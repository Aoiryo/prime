from enum import Enum
from typing import Any, Literal, TypeAlias
import os

from pydantic import create_model, model_validator
from pydantic_config import BaseConfig

AttnFnType: TypeAlias = Literal["flex", "math"]

class Compression(Enum):
    NO = "no"
    UINT8 = "uint8"


class DataConfig(BaseConfig):
    dataset_name_or_paths: str = "datasets/fineweb-edu"
    val_dataset_name_or_paths: str | None = None
    sequence_packing: bool = True
    seq_length: int = 1024
    fake: bool = False
    num_workers: int = 4
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    dataset_ratio: str | None = None
    data_rank: int | None = None
    data_world_size: int | None = None
    reverse_data_files: bool = False
    split_by_data_rank: bool = True


class AdamConfig(BaseConfig):
    type: Literal["adam"] = (
        "adam"  # the literal is used to distinguish between the different optimizers configuration in the union type
    )
    lr: float = 4e-4
    weight_decay: float = 0.1
    betas1: float = 0.9
    betas2: float = 0.95


class SoapConfig(BaseConfig):
    type: Literal["soap"] = "soap"
    lr: float = 4e-4
    weight_decay: float = 1e-05
    betas1: float = 0.9
    betas2: float = 0.95

    max_preconditioner_dim: int = 8192
    precondition_frequency: int = 100


OptimizersConfig: TypeAlias = AdamConfig | SoapConfig


class OptimConfig(BaseConfig):
    optim: OptimizersConfig = AdamConfig()

    sched_type: Literal["cosine", "linear", "wsd-sqrt"] = "cosine"
    warmup_steps: int = 1000
    stable_steps: int = 80_000
    total_steps: int = 88_000
    batch_size: int = 512

    z_loss: bool = False
    z_loss_weight: float = 2e-4
    num_chunks: int | None = None


class DilocoConfig(BaseConfig):
    outer_lr: float = 0.7
    inner_steps: int
    compression: Compression = Compression.NO

    retry_all_reduce: int = 3


class MemoryProfilerConfig(BaseConfig):
    freq: int = 10
    snapshot_dir: str


class TrainConfig(BaseConfig):
    micro_bs: int = 1

    ac_ckpt: bool | int = False
    reshard_after_forward: bool = True  # old shard grad op True mean full shard

    reduce_fp32: bool = False  # should be True if SXM. Keep to false as default for backward compatibility

    log_model_hash: bool = False

    memory_profiler: MemoryProfilerConfig | None = None

    torch_profiler: bool = False

    torch_compile: bool = True

    fused_linear_ce: bool = False

    fsdp_cpu_offload: bool = False

    attn_fn: AttnFnType = "flex"


class MonitorConfig(BaseConfig):
    log_flush_interval: int = 10
    base_url: str | None = None
    auth_token: str | None = None


class RepaConfig(BaseConfig):
    # logging params
    output_dir: str = "exps"
    exp_name: str
    logging_dir: str = "logs"
    report_to: str = "wandb"
    sampling_steps: int = 10000
    resume_step: int = 0
    continue_train_exp_dir: str | None = None
    wandb_history_path: str | None = None

    # SiT model params
    model: Literal["SiT-B/2", "SiT-L/2", "SiT-XL/2", "SiT-B/1", "SiT-L/1", "SiT-XL/1"] = "SiT-XL/2"
    num_classes: int = 1000
    encoder_depth: int = 8
    qk_norm: bool = False
    fused_attn: bool = True
    bn_momentum: float = 0.1
    compile: bool = True

    # dataset params
    data_dir: str = "data"
    resolution: Literal[256] = 256
    batch_size: int = 256

    # precision params
    allow_tf32: bool = True
    mixed_precision: Literal["no", "fp16", "bf16"] = "fp16"

    # optimization params
    epochs: int = 1400
    max_train_steps: int = 400000
    checkpointing_steps: int = 50000
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 0.0
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # seed params
    seed: int = 0

    # cpu params
    num_workers: int = 4

    # loss params
    path_type: Literal["linear", "cosine"] = "linear"
    prediction: Literal["v"] = "v"
    cfg_prob: float = 0.1
    enc_type: str = "dinov2-vit-b"
    proj_coeff: float = 0.5
    weighting: Literal["uniform", "lognormal"] = "uniform"

    # vae params
    vae: Literal["f8d4", "f16d32"] = "f8d4"
    vae_ckpt: str = "pretrained/sdvae-f8d4/sdvae-f8d4.pt"

    # vae loss params
    disc_pretrained_ckpt: str | None = None
    loss_cfg_path: str = "configs/l1_lpips_kl_gan.yaml"

    # vae training params
    vae_learning_rate: float = 1e-4
    disc_learning_rate: float = 1e-4
    vae_align_proj_coeff: float = 1.5


class RemoteConfig(BaseConfig):
    path: str  # could be a s3 path
    interval: int


class CkptConfig(BaseConfig):
    path: str | None = None
    interval: int | None = None
    topk: int | None = None

    remote: RemoteConfig | None = None

    remote_data_path: str | None = None
    remote_data_load: bool = False

    resume: str | None = None

    skip_dataloader: bool = False

    live_recovery_rank_src: int | None = None

    data_path: str | None = None

    token_count: int | None = None

    @model_validator(mode="after")
    def validate_path_and_interval(self):
        if (self.path is None) != (self.interval is None):
            raise ValueError("path and interval must be both set or both None")
        if self.path is None and self.remote is not None:
            raise ValueError("remote_path is set but path is not set")

        return self

    @model_validator(mode="after")
    def validate_remote_data_path(self):
        if self.remote_data_load and self.data_path is not None:
            raise ValueError("remote_data_load and data_path are mutually exclusive")

        if self.remote_data_load and self.remote_data_path is None:
            raise ValueError("remote_data_load is set but remote_data_path is not set")
        return self


ENV_VAR_PREFIX = "ZERO_BAND_"

class Config(BaseConfig):
    # main config
    name_model: Literal["debugmodel", "70M","150M", "271M", "1B", "7B", "10B", "13B", "26B", "70B"] = "150M"
    type_model: Literal["llama2", "llama3", "repa"] = "llama3"

    # Project/Run
    project: str = "zeroband"
    run_id: str | None = None
    run_name: str | None = None

    # Logger
    metric_logger_type: Literal["wandb", "dummy"] = "wandb"
    wandb_resume: bool = False
    log_level: Literal["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_all_rank: bool = False

    # sub config
    diloco: DilocoConfig | None = None
    data: DataConfig = DataConfig()
    optim: OptimConfig = OptimConfig()
    train: TrainConfig
    monitor: MonitorConfig | None = None
    repa: RepaConfig | None = None

    ckpt: CkptConfig = CkptConfig()

    @model_validator(mode="after")
    def ckpt_diloco_step(self):
        if self.ckpt is not None and self.ckpt.interval is not None and self.diloco is not None:
            assert (
                self.ckpt.interval % self.diloco.inner_steps == 0
            ), "ckpt interval must be a multiple of diloco inner steps as we only save at the end of an outer step"
        return self

    @model_validator(mode="after")
    def validate_live_recovery_rank_src(self):
        if self.ckpt is not None and self.ckpt.live_recovery_rank_src is not None and self.diloco is None:
            raise ValueError("live_recovery_rank_src is only supported with diloco")
        return self


def resolve_env_vars(config: Config) -> None:
    """
    Resolve environment variables for config fields.
    Modifies the config in place.
    Environment variables should be prefixed with ZERO_BAND_.
    """

    def _resolve_value(env_var: str, field_name: str, config_obj: Any) -> Any:
        """
        Resolve a single value from an environment variable
        env_var: full environment variable name (e.g. ZERO_BAND_TRAIN_MICRO_BS)
        field_name: actual field name in the config object (e.g. micro_bs)
        """
        value = os.environ.get(env_var)
        if value is not None:
            if (field_info := config_obj.__class__.model_fields.get(field_name)) is None:
                raise AttributeError(f"Config {config_obj} has no attribute {field_name}")

            try:
                # Create a temporary model with just this field, then validate and rip it out.
                py_model = create_model('TempModel', __base__ = BaseConfig, **{field_name: (field_info.annotation, ...)}) # type: ignore
                validated = py_model.model_validate({field_name: value})
                return getattr(validated, field_name)
            except Exception as e:
                raise ValueError(f"Error setting {env_var}={value}: {e}")
        return None

    def _resolve_nested(prefix: str, config_obj: Any) -> None:
        if not hasattr(config_obj, 'model_fields'):
            return

        for field_name, _ in config_obj.__class__.model_fields.items():
            # Build the full env var name
            full_env_var = f"{ENV_VAR_PREFIX}{prefix}_{field_name}".upper() if prefix else f"{ENV_VAR_PREFIX}{field_name}".upper()

            # Try to resolve the field directly using the local field name
            value = _resolve_value(full_env_var, field_name, config_obj)
            if value is not None:
                setattr(config_obj, field_name, value)

            # Handle nested configs
            field_value = getattr(config_obj, field_name)
            if field_value is not None and hasattr(field_value, 'model_fields'):
                # Pass the prefix for building env var names, but use local field names for lookup
                _resolve_nested(f"{prefix}_{field_name}" if prefix else field_name, field_value)

    def _get_valid_env_vars(prefix: str, config_obj: Any) -> set[str]:
        """Recursively collect all valid environment variable names"""
        valid_vars = set()
        if not hasattr(config_obj, 'model_fields'):
            return valid_vars

        for field_name, _ in config_obj.__class__.model_fields.items():
            full_env_var = f"{ENV_VAR_PREFIX}{prefix}_{field_name}".upper() if prefix else f"{ENV_VAR_PREFIX}{field_name}".upper()
            valid_vars.add(full_env_var)

            field_value = getattr(config_obj, field_name)
            if field_value is not None and hasattr(field_value, 'model_fields'):
                nested_prefix = f"{prefix}_{field_name}" if prefix else field_name
                valid_vars.update(_get_valid_env_vars(nested_prefix, field_value))

        return valid_vars

    # Check for any invalid ZERO_BAND_ environment variables
    valid_env_vars = _get_valid_env_vars("", config)
    invalid_vars = []
    for env_var in os.environ:
        if env_var.startswith(ENV_VAR_PREFIX) and env_var not in valid_env_vars:
            invalid_vars.append(env_var)

    if invalid_vars:
        raise ValueError(
            f"Found invalid environment variables with {ENV_VAR_PREFIX} prefix: {', '.join(invalid_vars)}\n"
             "See the full list of valid config veriables in src/zeroband/config.py."
        )

    # Now resolve the valid ones.
    _resolve_nested("", config)
