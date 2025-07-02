import re
import time
import torch
from torch import nn
from zeroband.comms import ElasticDeviceMesh
from zeroband.collectives import Compression, all_reduce
from zeroband.utils.world_info import get_world_info
from zeroband.utils.logger import get_logger
from zeroband.config import DilocoConfig
import torch.distributed as dist
from torch.distributed._tensor.api import DTensor
from functools import lru_cache


@lru_cache(maxsize=None)
def _find_first_number(s: str) -> int:
    match = re.search(r"\d+", s)
    if match:
        return int(match.group())
    else:
        return -1

class Diloco:
    def __init__(
        self,
        config: DilocoConfig,
        models: list[nn.Module],  # 接受模型列表
        elastic_device_mesh: ElasticDeviceMesh,
    ):
        self.config = config
        self.models = models
        self.elastic_device_mesh = elastic_device_mesh

        if config.compression == Compression.UINT8:
            from zeroband.C.collectives import ring_allreduce as _  # force compilation

        self._logger = get_logger()
        self.world_info = get_world_info()

        self._init_offloaded_optimizer()

    @torch.no_grad()
    def _init_offloaded_optimizer(self):
        # 对所有模型 offload param
        self.param_list_cpu = []
        self._offloaded_grad_grouped_tensor = []
        for model in self.models:
            offloaded_params, grouped_tensor = self.get_offloaded_param(model)
            self.param_list_cpu.extend(offloaded_params)
            self._offloaded_grad_grouped_tensor.extend(grouped_tensor)

        self.outer_optimizer = torch.optim.SGD(
            self.param_list_cpu, lr=self.config.outer_lr, momentum=0.9, nesterov=True
        )
        self._logger.debug("Offloaded all models to CPU")

    @torch.no_grad()
    def sync_pseudo_gradient(self, fake: bool = False, flag: str = "outer"):
        _start_time = time.perf_counter()
        self.elastic_device_mesh.maybe_reinit_global_pg(admit_joiners=False)
        world_size = self.elastic_device_mesh.global_pg.size()
        self._logger.debug("Sync pseudo gradient %s with world size %d", " fake" if fake else "", world_size)

        global_pg = self.elastic_device_mesh.global_pg
        param_idx = 0

        for i in range(self.config.retry_all_reduce):
            try:
                for model in self.models:
                    for param in model.parameters():
                        param_offloaded = self.param_list_cpu[param_idx]
                        assert isinstance(param_offloaded.grad, DTensor)
                        if fake:
                            param_offloaded.grad.to_local().zero_()
                        else:
                            param_offloaded.grad.to_local().copy_(param_offloaded.data.to_local())
                            param_offloaded.grad.to_local().sub_(param.data.to_local().to(param_offloaded.data.device))
                        param_idx += 1

                self.offloaded_grad_flat_tensor.div_(world_size)
                self._logger.debug("Waiting on barrier")
                self.elastic_device_mesh.monitored_barrier(flag)

                self._logger.debug("Beginning all reduce")
                for j, tensor_group in enumerate(self._offloaded_grad_grouped_tensor):
                    t0 = time.perf_counter()
                    all_reduce(self.config.compression, tensor_group, dist.ReduceOp.SUM, global_pg)
                    self._logger.debug(
                        f"{j}/{len(self._offloaded_grad_grouped_tensor)} all reduce bucket done in {time.perf_counter() - t0:.6f} seconds, numel: {tensor_group.numel()}"
                    )
                break
            except Exception as e:
                self._logger.error(f"Error syncing pseudo gradient: {e}, retry {i+1}/{self.config.retry_all_reduce}")
                global_pg = self.elastic_device_mesh.get_global_pg(maybe_reinit=True)
        else:
            self._logger.error("Failed to sync pseudo gradient after retries. Falling back to local calculation")
            param_idx = 0
            for model in self.models:
                for param in model.parameters():
                    param_offloaded = self.param_list_cpu[param_idx]
                    if fake:
                        param_offloaded.grad.to_local().zero_()
                    else:
                        param_offloaded.grad.to_local().copy_(param_offloaded.data.to_local())
                        param_offloaded.grad.to_local().sub_(param.data.to_local().to(param_offloaded.data.device))
                    param_idx += 1

        self._logger.info(f"Sync pseudo-gradient in {time.perf_counter() - _start_time:.6f} seconds")

    @torch.no_grad()
    def sync_inner_model(self):
        self._logger.debug("Sync inner models")
        param_idx = 0
        for model in self.models:
            for param in model.parameters():
                param_offloaded = self.param_list_cpu[param_idx]
                param.data.to_local().copy_(param_offloaded.data.to_local())
                param_idx += 1

    @torch.no_grad()
    def get_offloaded_param(self, model: nn.Module) -> list[nn.Parameter]:
        param_items = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
        numels = sum(param.to_local().numel() for _, param in param_items)

        self.offloaded_data_flat_tensor = torch.empty((numels,), device="cpu", dtype=torch.float32)
        self.offloaded_grad_flat_tensor = torch.zeros((numels,), device="cpu", dtype=torch.float32)
        current_offset = 0
        offloaded_params = []
        param_group_cutoff = []

        prev_id = None
        for name, param in param_items:
            if _find_first_number(name) != prev_id:
                param_group_cutoff.append(current_offset)
                prev_id = _find_first_number(name)

            target = param.data.to_local().detach()
            data_tensor = self.offloaded_data_flat_tensor.as_strided(target.size(), target.stride(), current_offset)
            grad_tensor = self.offloaded_grad_flat_tensor.as_strided(target.size(), target.stride(), current_offset)
            current_offset += data_tensor.numel()
            data_tensor.copy_(target)

            offloaded_param = nn.Parameter(
                DTensor.from_local(
                    data_tensor,
                    device_mesh=self.elastic_device_mesh.cpu_local_mesh,
                    placements=param.data.placements,
                )
            )

            offloaded_param.grad = DTensor.from_local(
                grad_tensor,
                device_mesh=self.elastic_device_mesh.cpu_local_mesh,
                placements=param.data.placements,
            )
            offloaded_param.requires_grad = True
            offloaded_params.append(offloaded_param)

        param_group_cutoff.append(current_offset)

        return offloaded_params, [
            self.offloaded_grad_flat_tensor.as_strided((j - i,), (1,), i)
            for i, j in zip(param_group_cutoff, param_group_cutoff[1:])
        ]

    @torch.no_grad()
    def step(self, fake: bool = False, flag: str = "outer"):
        time_start = time.perf_counter()
        self.sync_pseudo_gradient(fake=fake, flag=flag)
        self._logger.info(f"All reduce pseudo gradient in: {time.perf_counter() - time_start} seconds")

        if self.outer_optimizer is not None:
            self.outer_optimizer.step()

        self.sync_inner_model()
