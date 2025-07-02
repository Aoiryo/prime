import os
import sys
import time
import signal
from functools import partial
from typing import TYPE_CHECKING
from multiprocessing.process import _children  # type: ignore

import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy  # type: ignore
from torch.autograd.profiler import record_function

from zeroband.checkpoint import CkptManager, TrainingProgress
from zeroband.comms import ElasticDeviceMesh
from zeroband.config import Config, resolve_env_vars
from zeroband.data import TEST_VOCAB_SIZE, get_dataloader
from zeroband.diloco import Diloco
from zeroband.loss import compute_cross_entropy_loss
from zeroband.lr_scheduler import get_scheduler
from zeroband.models.llama import get_model
from zeroband.optimizers import get_optimizer
from zeroband.utils import (
    FakeTokenizer,
    PerfCounter,
    get_module_signature,
    get_optimizer_signature,
    get_tensor_list_signature,
    get_peak_flops,
    get_num_params,
    get_num_flop_per_token,
)
from zeroband.utils.metric_logger import MetricLogger, WandbMetricLogger, DummyMetricLogger
from zeroband.utils.activation_ckpt import apply_ac_ckpt
from zeroband.utils.profiler import MemoryProfiler
from zeroband.utils.world_info import get_world_info
from zeroband.utils.logger import get_logger
from zeroband.utils.stopwatch import Stopwatch

from transformers import AutoTokenizer
from pydantic_config import parse_argv
import argparse

from dataset import CustomINH5Dataset
from utils import load_encoders, normalize_latents, denormalize_latents, preprocess_imgs_vae, count_trainable_params
from copy import deepcopy
from collections import OrderedDict


def sigterm_handler(signum, frame):
    print(f"[Rank {os.environ.get('RANK', '?')}] Returning exit code 1...")
    sys.exit(1)
    

def log_hash_training_state(
    config: Config,
    model: torch.nn.Module,
    inner_optimizer: torch.optim.Optimizer,
    diloco: Diloco | None,
    metric_logger: MetricLogger | None,
    step: int,
    id: str = "",
):
    """Log the hash of the model and optimizer. This function is slow"""
    if config.train.log_model_hash:
        inner_model_hash = get_module_signature(model)
        inner_optimizer_hash = get_optimizer_signature(inner_optimizer)

        logger.debug(f"inner diloco model {id} : {inner_model_hash}")
        logger.debug(f"inner optimizer hash {id} : {inner_optimizer_hash}")

        metrics = {
            "step": step,
            f"inner_model_hash_{id}": inner_model_hash,
            f"inner_optimizer_hash_{id}": inner_optimizer_hash,
        }

        if config.diloco is not None and diloco is not None:
            outer_optimizer_hash = get_optimizer_signature(diloco.outer_optimizer)
            outer_model_hash = get_tensor_list_signature(diloco.param_list_cpu)  # type: ignore

            logger.debug(f"outer diloco optimizer hash {id} : {outer_optimizer_hash}")
            logger.debug(f"outer diloco model hash {id} : {outer_model_hash}")

            metrics.update(
                {f"outer_optimizer_hash_{id}": outer_optimizer_hash, f"outer_model_hash_{id}": outer_model_hash}
            )
        if world_info.rank == 0:
            assert metric_logger is not None
            metric_logger.log(metrics)


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

    # Also perform EMA on BN buffers
    ema_buffers = OrderedDict(ema_model.named_buffers())
    model_buffers = OrderedDict(model.named_buffers())

    for name, buffer in model_buffers.items():
        name = name.replace("module.", "")
        if buffer.dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
            # Apply EMA only to float buffers
            ema_buffers[name].mul_(decay).add_(buffer.data, alpha=1 - decay)
        else:
            # Direct copy for non-float buffers
            ema_buffers[name].copy_(buffer)


def train(config: Config, args = None):

    # signal.signal(signal.SIGTERM, sigterm_handler)

    # batch_size is the total batch size for all GPUs
    assert config.optim.batch_size % world_info.local_world_size == 0
    batch_size = config.optim.batch_size // world_info.local_world_size

    assert batch_size % config.train.micro_bs == 0, (
        f"The micro batch size ({config.train.micro_bs}) must divide the number of samples on each GPU ({batch_size})."
    )
    # REPA use a default of 1 grad accu step
    gradient_accumulation_steps = args.gradient_accumulation_steps

    if config.ckpt is not None and config.ckpt.interval is not None and config.diloco is not None:
        assert config.ckpt.interval % config.diloco.inner_steps == 0, (
            "ckpt interval must be a multiple of diloco inner steps as we only save at the end of an outer step"
        )

    sw = Stopwatch(config)
    sw.start("train()")

    # Load tokenizer
    with sw.record_block("Load Tokenizer"):
        if config.data.fake and config.name_model == "debugmodel":
            tokenizer = FakeTokenizer()
        elif config.type_model == "llama2":
            tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True)
        elif config.type_model == "llama3":
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_fast=True)
        elif config.type_model == "repa":
            # REPA is a diffusion model, no need for a tokenizer
            pass
        else:
            raise ValueError(f"Model type {config.type_model} not supported")

    with sw.record_block("Get Dataloader"):
        if config.type_model != "repa":
            train_dataloader = get_dataloader(
                tokenizer=tokenizer,
                world_size=world_info.world_size,
                rank=world_info.rank,
                batch_size=config.train.micro_bs,
                data_config=config.data,
            )
            train_dataloader_iterator = iter(train_dataloader)
        else:
            train_dataset = CustomINH5Dataset(args.data_dir)

            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_info.world_size,
                rank=rank,
                shuffle=True
            )

            local_batch_size = args.batch_size // world_info.world_size
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=local_batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True
            ) # TODO: support micro bs, but for now we stick to the original handling from RE
            train_dataloader_iterator = iter(train_dataloader)

    with sw.record_block("Get Model"):
        if config.type_model != "repa":
            model, model_config = get_model(
                config,
                vocab_size=len(tokenizer) if config.name_model != "debugmodel" or not config.data.fake else TEST_VOCAB_SIZE,
            )
        else:
            if args.vae == "f8d4":
                assert args.resolution % 8 == 0, "Image size must be divisible by 8"
                latent_size = args.resolution // 8
                in_channels = 4
            elif args.vae == "f16d32":
                assert args.resolution % 16 == 0, "Image size must be divisible by 16"
                latent_size = args.resolution // 16
                in_channels = 32
            else:
                raise NotImplementedError()

            # load encoders
            if args.enc_type != None:
                encoders, encoder_types, architectures = load_encoders(
                    args.enc_type, device, args.resolution
                )
            else:
                raise NotImplementedError()
            z_dims = [encoder.embed_dim for encoder in encoders] if args.enc_type != 'None' else [0]

            # SiT model
            block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
            model = SiT_models[args.model](
                input_size=latent_size,
                in_channels=in_channels,
                num_classes=args.num_classes,
                class_dropout_prob=args.cfg_prob,
                z_dims=z_dims,
                encoder_depth=args.encoder_depth,
                bn_momentum=args.bn_momentum,
                **block_kwargs
            ).to(device)

            # ema model
            ema = copy.deepcopy(model).to(device)
            requires_grad(ema, False)

            # load VAE ckpt
            vae = vae_models[args.vae]().to(device)
            vae_ckpt = torch.load(args.vae_ckpt, map_location=device)
            vae.load_state_dict(vae_ckpt, strict=False)
            del vae_ckpt

            # init bn
            latents_stats = torch.load(args.vae_ckpt.replace(".pt", "-latents-stats.pt"))
            latents_scale = latents_stats["latents_scale"].squeeze().to(device)
            latents_bias = latents_stats["latents_bias"].squeeze().to(device)
            model.init_bn(latents_bias=latents_bias, latents_scale=latents_scale)

            loss_cfg = OmegaConf.load(args.loss_cfg_path)
            vae_loss_fn = ReconstructionLoss_Single_Stage(loss_cfg).to(device) # TODO: put loss cfg to the path and set

            update_ema(ema, model, decay=0)

    gpu_peak_flops = get_peak_flops(torch.cuda.get_device_name(torch.device("cuda")))
    logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")

    num_params = get_num_params(model, exclude_embedding=True)
    logger.info(f"Number of parameters: {num_params}")
    num_flop_per_token = get_num_flop_per_token(
        num_params,
        model_config,
        config.data.seq_length,
    )

    with sw.record_block("Shard Model"):
        if config.train.ac_ckpt:
            num = 1 if isinstance(config.train.ac_ckpt, bool) else config.train.ac_ckpt
            apply_ac_ckpt(model, num)

        elastic_device_mesh = ElasticDeviceMesh(
            enable=config.diloco is not None, live_recovery_rank_src=config.ckpt.live_recovery_rank_src
        )

        elastic_device_mesh.cuda_local_mesh = elastic_device_mesh.mesh["intranode"]
        elastic_device_mesh.cuda_local_mesh._dim_group_infos = []
        elastic_device_mesh.cuda_local_mesh._dim_group_infos.append(elastic_device_mesh.mesh._dim_group_infos[-1])

        elastic_device_mesh.cpu_local_mesh = elastic_device_mesh.mesh["intranode"]
        elastic_device_mesh.cpu_local_mesh._dim_group_infos = []
        elastic_device_mesh.cpu_local_mesh._dim_group_infos.append(elastic_device_mesh.mesh._dim_group_infos[-1])
        
        # signal.signal(signal.SIGTERM, partial(lambda edm, signum, frame: sigterm_handler(edm), elastic_device_mesh))
        
        # from ipdb import set_trace; set_trace()
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32 if config.train.reduce_fp32 else None
        )

        offload_policy = CPUOffloadPolicy(pin_memory=True) if config.train.fsdp_cpu_offload else None

        fully_shard(
            vae,
            mp_policy=mp_policy,
            mesh=elastic_device_mesh.cuda_local_mesh,
            reshard_after_forward=config.train.reshard_after_forward,
            offload_policy=offload_policy,
        )

        fully_shard(
            model,
            mp_policy=mp_policy,
            mesh=elastic_device_mesh.cuda_local_mesh,
            reshard_after_forward=config.train.reshard_after_forward,
            offload_policy=offload_policy,
        )

        fully_shard(
            vae_loss_fn,
            mp_policy=mp_policy,
            mesh=elastic_device_mesh.cuda_local_mesh,
            reshard_after_forward=config.train.reshard_after_forward,
            offload_policy=offload_policy,
        )

    # Setup optimizers
    with sw.record_block("Optimizer Setup"):

        optimizer_vae = torch.optim.AdamW(
            vae.parameters(),
            lr=args.vae_learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        
        optimizer_loss_fn = torch.optim.AdamW(
            vae_loss_fn.parameters(),
            lr=args.disc_learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        diloco = Diloco(config.diloco, [vae, model, vae_loss_fn], elastic_device_mesh) if config.diloco is not None else None

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0) # TODO: change this later

        training_progress = TrainingProgress(total_tokens=0, outer_step=0, step=0)

        ckpt_manager = CkptManager(
            config=config.ckpt,
            model={"vae": vae, "model": model, "vae_loss_fn": vae_loss_fn},
            optimizer={"vae": optimizer_vae, "model": optimizer, "vae_loss_fn": optimizer_loss_fn},
            scheduler=scheduler,
            dataloader=train_dataloader,
            training_progress=training_progress,
            data_rank=config.data.data_rank,
            diloco_offloaded_optimizer=diloco.outer_optimizer if config.diloco is not None else None,  # type: ignore
            diloco_offloaded_param_list=diloco.param_list_cpu if config.diloco is not None else None,  # type: ignore
        )

    if world_info.rank == 0:
        logger_cls = WandbMetricLogger if config.metric_logger_type == "wandb" else DummyMetricLogger
        metric_logger = logger_cls(
            project=config.project,
            logger_config={"config": config.model_dump(), "world_info": world_info.json()},
            resume=config.wandb_resume,
        )
    else:
        metric_logger = None

    with sw.record_block("Compile Model"):
        if config.train.torch_compile:
            # we need to compile AFTER creating the CKPT manager, DON'T ASK ME WHY
            model = torch.compile(model, backend="inductor", mode="default")
            vae = torch.compile(vae, backend="inductor", mode="default")
            vae_loss_fn = torch.compile(vae_loss_fn, backend="inductor", mode="default")

    # if config.ckpt.resume is not None:
    #     with sw.record_block("Resume Checkpoint"):
    #         # all is inplace
    #         ckpt_manager.load(
    #             resume_ckpt_path=config.ckpt.resume,
    #             skip_dataloader=config.ckpt.skip_dataloader,
    #             data_path=config.ckpt.data_path,
    #         )
    #         log_hash_training_state(
    #             config, model, inner_optimizer, diloco, metric_logger, step=training_progress.step, id="resume"
    #         )

    if config.train.memory_profiler is not None:
        memory_profiler = MemoryProfiler(config.train.memory_profiler.freq, config.train.memory_profiler.snapshot_dir)

    num_inner_steps = config.diloco.inner_steps if config.diloco is not None else 1
    perf_counter = PerfCounter(window_size=10)

    logger.debug("Finished setup in %f seconds", sw.elapsed())

    need_live_recovery = config.ckpt.live_recovery_rank_src is not None
    while True: # control the training loop with step
        model.train()
        if num_inner_steps > 1:
            # if we don't use diloco we don't print the outer step logs
            logger.info(f"outer_step step: {training_progress.outer_step}")

        time_start_outer = time.perf_counter()

        if config.diloco is not None:
            assert diloco is not None
            # this is a patch for now to allow live recovery worker to not affect the all reduce at all

            if not need_live_recovery:
                elastic_device_mesh.maybe_reinit_global_pg(admit_joiners=True)

                maybe_dest_rank = elastic_device_mesh.live_recovery.should_send_ckpt_to()
                if maybe_dest_rank is not None:
                    logger.info(f"Start live recovery to rank {maybe_dest_rank}")
                #     ckpt_manager.send_ckpt_to_peer(elastic_device_mesh.global_pg, maybe_dest_rank, blocking=True)

                    # TODO: block until the worker loads ckpt successfully
                    elastic_device_mesh.global_store.get("live_recovery_loaded")

                    elastic_device_mesh.live_recovery.reset()
                    elastic_device_mesh.global_store.delete_key("live_recovery_loaded")

            else:
                ## receiving
                time_start_live_recovery = time.perf_counter()
                logger.info(f"Start live recovery from rank {config.ckpt.live_recovery_rank_src}")

                ## we create grad buffer and opts stats mamnually, the value will be overwritten by the ckpt but we need the DTensor to be correctly init before loading it

                diloco.outer_optimizer.step()  # need to step to init the DTensor stats

                # ckpt_manager.recv_ckpt_from_peer(elastic_device_mesh.global_pg)

                # TODO: 1. make sure that the upload is successful and atomic
                # time.sleep(15)
                # this is done through tcp store get blocking
                
                # TODO: 2. use ckpt.load to resume
                if config.ckpt.resume:
                    while True:
                        upload_status = elastic_device_mesh.god_store.get("upload_successful").decode("utf-8")
                        if upload_status == "stable":
                            break
                        else:
                            logger.info(f"upload status: {upload_status}")
                            time.sleep(5)
                    print(f"{world_info.local_rank} rank check before ckpt manager")
                    ckpt_manager.load(
                        resume_ckpt_path=config.ckpt.resume,
                        skip_dataloader=config.ckpt.skip_dataloader,
                        data_path=config.ckpt.data_path,
                        group=elastic_device_mesh.ckpt_pg,
                    )
                # TODO: 3: tell the master to be free
                elastic_device_mesh.global_store.set("live_recovery_loaded", "success")

                log_hash_training_state(
                    config,
                    model,
                    inner_optimizer,
                    diloco,
                    metric_logger,
                    step=training_progress.step,
                    id="live_reco_recv",
                )
                need_live_recovery = False

                if config.ckpt.remote_data_load:
                    ckpt_manager.remote_data_load()

                logger.info("live recovery done in %f", time.perf_counter() - time_start_live_recovery)

        # at the beginning of the inner steps we allow joiner to arrive.
        # We maybe reinit before the all reduce but only to allow leaving, not to join anymore

        for inner_step in range(num_inner_steps):
            logger.debug("Starting inner step.")
            sw.start("inner_step")

            loss_batch = 0
            z_loss_batch = 0

            with sw.record_block("Grad Acc Steps"):
                for grad_acc_step in range(gradient_accumulation_steps):
                    sw.start("grad_acc_step")

                    is_accumulating = grad_acc_step < gradient_accumulation_steps - 1
                    # no sync if we are accumulating gradients
                    model.set_requires_gradient_sync(not is_accumulating)

                    with sw.record_block("Load batch"):
                        # TODO/NOTE: We could overlap sending the batch with communication
                        #            although to be honest the perf impact is minimal
                        batch = next(train_dataloader_iterator) # raw image and y (label) for REPA
                        if config.type_model != "repa":
                            input_ids = batch["input_ids"]
                            labels = batch["labels"]
                            block_mask = batch["block_mask"]
                        else:
                            raw_image = batch["raw_image"]
                            labels = batch["labels"]

                    with sw.record_block("Run forward()"):
                        if config.type_model != "repa":
                            logits = model(tokens=input_ids, block_mask=block_mask).contiguous()
                            flatten_logits = logits.reshape(-1, logits.size(-1))  # b seq vocab -> (b * seq) vocab
                            flatten_labels = labels.reshape(-1)  # b seq -> (b * seq)
                        else:
                            with sw.record_block("REPA forwarding and loss calc"):
                                z = None
                                # extract the dinov2 features
                                with torch.no_grad():
                                    zs = []
                                    for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
                                        raw_image_ = preprocess_raw_image(raw_image, encoder_type)
                                        z = encoder.forward_features(raw_image_)
                                        if 'mocov3' in encoder_type: 
                                            z = z[:, 1:] 
                                        if 'dinov2' in encoder_type: 
                                            z = z['x_norm_patchtokens']
                                        zs.append(z)
                            
                                vae.train()
                                model.train()

                                processed_image = preprocess_imgs_vae(raw_image)
                                posterior, z, recon_image = vae(processed_image)

                                loss_kwargs = dict(
                                    path_type=args.path_type,
                                    prediction=args.prediction,
                                    weighting=args.weighting,
                                )
                                # Record the time_input and noises for the VAE alignment, so that we avoid sampling again
                                time_input = None
                                noises = None

                                # Turn off grads for the SiT model (avoid REPA gradient on the SiT model)
                                requires_grad(model, False)
                                # Avoid BN stats to be updated by the VAE
                                model.eval()

                                vae_loss, vae_loss_dict = vae_loss_fn(processed_image, recon_image, posterior, global_step, "generator")
                                vae_loss = vae_loss.mean()

                                # Compute the REPA alignment loss for VAE updates
                                loss_kwargs["align_only"] = True
                                vae_align_outputs = model(
                                    x=z,
                                    y=labels,
                                    zs=zs,
                                    loss_kwargs=loss_kwargs,
                                    time_input=time_input,
                                    noises=noises,
                                )
                                vae_loss = vae_loss + args.vae_align_proj_coeff * vae_align_outputs["proj_loss"].mean()

                                time_input = vae_align_outputs["time_input"]
                                noises = vae_align_outputs["noises"]

                                (vae_loss / args.gradient_accumulation_steps).backward()

                                if args.max_grad_norm > 0:
                                    grad_norm_vae = torch.nn.utils.clip_grad_norm_(vae.parameters(), args.max_grad_norm)
                                optimizer_vae.step()
                                optimizer_vae.zero_grad()

                                d_loss, d_loss_dict = vae_loss_fn(processed_image, recon_image, posterior, global_step, "discriminator")
                                d_loss = d_loss.mean()
                                (d_loss / args.gradient_accumulation_steps).backward()

                                if args.max_grad_norm > 0:
                                    grad_norm_disc = torch.nn.utils.clip_grad_norm_(vae_loss_fn.parameters(), args.max_grad_norm)
                                optimizer_loss_fn.step()
                                optimizer_loss_fn.zero_grad()

                                requires_grad(model, True)
                                model.train()

                                # Forward the SiT model
                                loss_kwargs["weighting"] = args.weighting
                                loss_kwargs["align_only"] = False
                                sit_outputs = model(
                                    x=z.detach(),
                                    y=labels,
                                    zs=zs,
                                    loss_kwargs=loss_kwargs,
                                    time_input=time_input,
                                    noises=noises,
                                )

                                # compute diffusion loss and REPA alignment loss, backpropagate the SiT loss, and update the model
                                sit_loss = sit_outputs["denoising_loss"].mean() + args.proj_coeff * sit_outputs["proj_loss"].mean()
                                (sit_loss / args.gradient_accumulation_steps).backward()

                                if args.max_grad_norm > 0:
                                    grad_norm_sit = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                                optimizer.step()
                                optimizer.zero_grad()

                                update_ema(ema, model, decay=0)

                                inner_optimizer = None # pass compilation check

                    if config.type_model != "repa":
                        with sw.record_block("Loss Calculation"):
                            ce_loss, z_loss = compute_cross_entropy_loss(
                                flatten_logits,
                                flatten_labels,
                                z_weight=config.optim.z_loss_weight if config.optim.z_loss else None,
                                num_chunks=config.optim.num_chunks,
                                fused_linear_weight=model.output.weight if config.train.fused_linear_ce else None,
                            )

                            del logits
                            del flatten_logits
                            del flatten_labels

                            if config.optim.z_loss:
                                assert z_loss is not None
                                ce_loss /= gradient_accumulation_steps
                                z_loss /= gradient_accumulation_steps
                                loss = ce_loss + z_loss
                            else:
                                loss = ce_loss / gradient_accumulation_steps

                        with sw.record_block("Run backward()"):
                            loss.backward()

                        with record_function("Clone Loss"):
                            # No need to time, takes 0 seconds
                            if config.optim.z_loss:
                                assert z_loss is not None
                                loss_batch += ce_loss.detach().clone()
                                z_loss_batch += z_loss.detach().clone()
                            else:
                                loss_batch += loss.detach().clone()

                        elapsed = sw.stop("grad_acc_step")
                        logger.debug(f"Grad acc step {grad_acc_step} completed in {elapsed:.2f} seconds")

            with sw.record_block("Loss allreduce()"):
                if config.type_model != "repa":
                    # Launch both allreduces at the same time to hide latency
                    loss_allreduce = dist.all_reduce(
                        tensor=loss_batch, op=dist.ReduceOp.AVG, group=elastic_device_mesh.local_pg, async_op=True
                    )
                    if config.optim.z_loss:
                        z_loss_allreduce = dist.all_reduce(
                            tensor=z_loss_batch, op=dist.ReduceOp.AVG, group=elastic_device_mesh.local_pg, async_op=True
                        )

                    assert isinstance(loss_allreduce, torch.distributed.Work)
                    loss_allreduce.wait()
                    if config.optim.z_loss:
                        assert isinstance(z_loss_allreduce, torch.distributed.Work)
                        z_loss_allreduce.wait()
                else:
                    vae_loss_allreduce = dist.all_reduce(
                        tensor=vae_loss, op=dist.ReduceOp.AVG, group=elastic_device_mesh.local_pg, async_op=True
                    )
                    d_loss_allreduce = dist.all_reduce(
                        tensor=d_loss, op=dist.ReduceOp.AVG, group=elastic_device_mesh.local_pg, async_op=True
                    )
                    sit_loss_allreduce = dist.all_reduce(
                        tensor=sit_loss, op=dist.ReduceOp.AVG, group=elastic_device_mesh.local_pg, async_op=True
                    )
                    assert isinstance(vae_loss_allreduce, torch.distributed.Work)
                    vae_loss_allreduce.wait()
                    assert isinstance(d_loss_allreduce, torch.distributed.Work)
                    d_loss_allreduce.wait()
                    assert isinstance(sit_loss_allreduce, torch.distributed.Work)
                    sit_loss_allreduce.wait()

            with sw.record_block("Clip Grad"):
                if config.type_model != "repa":
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).full_tensor()  # type: ignore (is a dtensor)
                # TODO: maybe add grad clip

            with sw.record_block("Optimizer Step"):
                if config.type_model != "repa":
                    inner_optimizer.step()
                    scheduler.step()

            with sw.record_block("Optimizer Zero Grad"):
                if config.type_model != "repa":
                    inner_optimizer.zero_grad()

            # logging
            training_progress.step += 1
            vae_inner_lr = [group["lr"] for group in optimizer_vae.param_groups][0]
            sit_inner_lr = [group["lr"] for group in optimizer.param_groups][0]
            loss_inner_lr = [group["lr"] for group in optimizer_loss_fn.param_groups][0]

            # syncing loss across all data parallel rank within a nodes
            new_tokens = config.data.seq_length * config.optim.batch_size
            perf_counter.count_tokens(new_tokens)

            if config.diloco is None:
                training_progress.total_tokens += new_tokens
            else:
                # we count the total tokens with respect to all diloco workers
                # might need to tweak this as some worker might fail to join the all reduce later
                training_progress.total_tokens += new_tokens * elastic_device_mesh.global_pg.size()

            assert isinstance(vae_loss, torch.Tensor)
            assert isinstance(d_loss, torch.Tensor)
            assert isinstance(sit_loss, torch.Tensor)
            
            metrics = {
                "vae_loss": vae_loss.item(),
                "d_loss": d_loss.item(),
                "sit_loss": sit_loss.item(),
                "step": training_progress.step,
                "vae_inner_lr": vae_inner_lr,
                "sit_inner_lr": sit_inner_lr,
                "loss_inner_lr": loss_inner_lr,
                "time": time.time(),
            }

            log = f"step: {training_progress.step}, loss: {loss_batch.item():.4f}"

            tokens_per_second = perf_counter.get_tokens_per_second()
            if tokens_per_second is not None:
                metrics["tokens_per_second"] = tokens_per_second
                metrics["mfu"] = (
                    100 * num_flop_per_token * tokens_per_second / gpu_peak_flops / world_info.local_world_size
                )
                log += f", tokens_per_second: {tokens_per_second:.2f}, mfu: {metrics['mfu']:.2f}"

            if config.diloco is not None:
                metrics["num_peers"] = elastic_device_mesh.global_pg.size()
                log += f", diloco_peers: {metrics['num_peers']}"

            if world_info.rank == 0:
                assert metric_logger is not None
                metric_logger.log(metrics)

            logger.info(log)

            if config.train.memory_profiler is not None:
                memory_profiler.step()

            elapsed = sw.stop("inner_step")
            logger.debug(f"Inner step {inner_step} completed in {elapsed:.2f} seconds")

        if config.diloco is not None:
            assert diloco is not None
            time_start_inner = time.perf_counter()
            diloco.step(flag=str(training_progress.outer_step))
            diloco_time = time.perf_counter() - time_start_inner

            log_hash_training_state(
                config, model, inner_optimizer, diloco, metric_logger, step=training_progress.step, id="outer_step"
            )

        training_progress.outer_step += 1

        if (
            config.ckpt.interval is not None
            and training_progress.step > 0
            and training_progress.step % config.ckpt.interval == 0
        ):
            # we only allow to checkpoint after a outer step. For non diloco training outer step = 1 anyway

            do_remote = config.ckpt.remote is not None and training_progress.step % config.ckpt.remote.interval == 0
            ckpt_manager.save(remote=do_remote, group=elastic_device_mesh.ckpt_pg, store=elastic_device_mesh.god_store)
            log_hash_training_state(
                config, model, inner_optimizer, diloco, metric_logger, step=training_progress.step, id="save"
            )

        if config.diloco:
            tokens_per_second = (
                config.optim.batch_size
                * config.diloco.inner_steps
                * config.data.seq_length
                / (time.perf_counter() - time_start_outer)
            )
            mfu = 100 * num_flop_per_token * tokens_per_second / gpu_peak_flops / world_info.local_world_size
            logger.info(f"effective mfu: {mfu}")

            if world_info.rank == 0:
                assert metric_logger is not None
                metric_logger.log(
                    {
                        "outer_mfu": mfu,
                        "step": training_progress.step,
                        "outer_step": training_progress.outer_step,
                        "outer_tokens_per_second": tokens_per_second,
                        "all_reduce_step": diloco_time,
                    }
                )

        if training_progress.step >= config.optim.total_steps:
            # we only allow to break outisde of the inner loop.
            # This avoid ending the training in the middle of a the inner loop
            # Since ckpt strategy and all reduce is done at the outer loop level.
            break

    if world_info.rank == 0:
        assert metric_logger is not None
        metric_logger.finish()

    ckpt_manager.wait_for_blocking_job()

    del elastic_device_mesh  # allow to clean up for smoother tests transition

    if config.train.memory_profiler is not None:
        logger.debug(f"Max memory used: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

    logger.info("Training finished, exiting ...")


if __name__ == "__main__":
    # Allow eager fallback during production so that that the training runs dont die
    # However, in development, we want to know that we broke torch compile
    torch._dynamo.config.suppress_errors = "ZERO_BAND_DEV" not in os.environ  # type: ignore
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(42)

    config = Config(**parse_argv())  # type: ignore
    resolve_env_vars(config)
    world_info = get_world_info()
    logger = get_logger(config)

    parser = argparse.ArgumentParser(description="Training")

    # logging params
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--sampling-steps", type=int, default=10000)
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--continue-train-exp-dir", type=str, default=None)
    parser.add_argument("--wandb-history-path", type=str, default=None)

    # SiT model params
    parser.add_argument("--model", type=str, default="SiT-XL/2", choices=SiT_models.keys(),
                        help="The model to train.")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bn-momentum", type=float, default=0.1)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                        help="Whether to compile the model for faster training")

    # dataset params
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--resolution", type=int, choices=[256], default=256)
    parser.add_argument("--batch-size", type=int, default=256)

    # precision params
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization params
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--checkpointing-steps", type=int, default=50000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    # seed params
    parser.add_argument("--seed", type=int, default=0)

    # cpu params
    parser.add_argument("--num-workers", type=int, default=4)

    # loss params
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"],
                        help="currently we only support v-prediction")
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-b')
    parser.add_argument("--proj-coeff", type=float, default=0.5)
    parser.add_argument("--weighting", default="uniform", type=str, choices=["uniform", "lognormal"],
                        help="Loss weihgting, uniform or lognormal")

    # vae params
    parser.add_argument("--vae", type=str, default="f8d4", choices=["f8d4", "f16d32"])
    parser.add_argument("--vae-ckpt", type=str, default="pretrained/sdvae-f8d4/sdvae-f8d4.pt")

    # vae loss params
    parser.add_argument("--disc-pretrained-ckpt", type=str, default=None)
    parser.add_argument("--loss-cfg-path", type=str, default="configs/l1_lpips_kl_gan.yaml")

    # vae training params
    parser.add_argument("--vae-learning-rate", type=float, default=1e-4)
    parser.add_argument("--disc-learning-rate", type=float, default=1e-4)
    parser.add_argument("--vae-align-proj-coeff", type=float, default=1.5)

    args = parser.parse_args()

    # torch.set_default_device("cuda")
    torch.cuda.set_device(world_info.local_rank)

    def pretty_dict(d, indent=2):
        for key, value in d.items():
            if isinstance(value, dict):
                logger.debug(" " * indent + f"{key}:")
                pretty_dict(value, indent + 2)
            else:
                logger.debug(" " * indent + f"{key}: {value}")

    logger.debug("config:")
    pretty_dict(config.model_dump())

    try:
        if config.train.torch_profiler and world_info.rank == 0:
            # NOTE(apaz-cli): I cannot seem to get the memory profiler to work.
            # Running into this issue: https://github.com/pytorch/pytorch/issues/64345
            # In the meantime, we can use the memory snapshotter.

            logger.debug("Running train() with profiler.")
            prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                # profile_memory=True,
                # with_stack=True,
            )
            try:
                prof.__enter__()
                train(config)
            finally:
                logger.debug("Exiting profiler context.")
                prof.__exit__(None, None, None)

            logger.info("Exporting chrome trace.")
            prof.export_chrome_trace("logs/profile.json.gz")

            width = 30
            logger.info("\n" + "*" * width + " GPU TIME " + "*" * width)
            logger.info(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

            logger.info("\n" + "*" * width + " GPU MEM " + "*" * width)
            logger.info(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

            # logger.info("Exporting memory timeline.")
            # prof.export_memory_timeline(f"logs/mem_timeline.html", device="cuda:0")
        else:
            train(config, args)
    except Exception as e:
        # Subprocesses can prevent the main process from exiting, so we need to terminate them
        logger.info("Caught an exception, terminating children")
        logger.info(e)
        for p in _children:
            p.terminate()

        raise e
