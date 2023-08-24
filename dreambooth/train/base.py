import os
import tempfile
from abc import ABC, abstractmethod
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Optional, Type, TypeVar

import diffusers.utils
import torch
import torch._dynamo
import torch._dynamo.config
import torch._dynamo.config_utils
import torch._functorch.config
import torch._inductor.config
import torch.backends.cuda
import torch.backends.cudnn
import torch.distributed
import torch.distributed.elastic.multiprocessing.errors
import torch.jit
import transformers.utils.logging
from accelerate.logging import get_logger
from accelerate.utils import InitProcessGroupKwargs
from diffusers import (
    AutoencoderKL,
    EulerDiscreteScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)

from dreambooth.params import Class, HyperParams, Model
from dreambooth.train.accelerators import BaseAccelerator
from dreambooth.train.eval import Evaluator
from dreambooth.train.shared import hash_bytes, images, main_process_only
from dreambooth.train.test import Tester

T = TypeVar("T")


class BaseTrainer(ABC):
    def __init__(self, *, instance_class: Class, params: HyperParams):
        self.instance_class = instance_class
        self.params = params

        self.priors_dir = Path(tempfile.mkdtemp())
        self.cache_dir = Path(os.getenv("CACHE_DIR", tempfile.mkdtemp()))
        self.metrics_cache = {}

        self.accelerator = self._accelerator()
        self.logger = get_logger(__name__)
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        self.tester = Tester(self.params, self.accelerator, instance_class)

        self.logger.warning(self.accelerator.state, main_process_only=False)
        self.logger.warning(
            f"Available GPU memory: {get_mem():.2f} GB", main_process_only=True
        )
        self.logger.warning(self.params.dict(), main_process_only=True)

        self._total_steps = 0

    def _accelerator(self) -> BaseAccelerator:
        try:
            from dreambooth.train.accelerators.colossal import (
                ColossalAccelerator as Accelerator,
            )
        except Exception:
            print("ColossalAI not installed, using default Accelerator")
            from dreambooth.train.accelerators.hf import HFAccelerator as Accelerator

        return Accelerator(
            params=self.params,
            mixed_precision=os.getenv("ACCELERATE_MIXED_PRECISION", "fp16"),
            log_with=["wandb"],
            gradient_accumulation_steps=self.params.gradient_accumulation_steps,
            kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(minutes=5))],
        )

    def _spawn(
        self, klass: Type[T], tap: Callable[[T], Any] = lambda x: x, **kwargs
    ) -> T:
        instance = klass.from_pretrained(
            self.params.model.name,
            revision=self.params.model.revision,
            local_files_only=True,
            **kwargs,
        )
        tap(instance)
        return instance

    def _vae(self, **kwargs):
        if self.params.model.vae:
            vae = AutoencoderKL.from_pretrained(
                self.params.model.vae,
                local_files_only=True,
                **kwargs,
            )
        else:
            vae = self._spawn(
                AutoencoderKL,
                subfolder="vae",
                **kwargs,
            )

        vae.requires_grad_(False)
        vae.enable_xformers_memory_efficient_attention()
        return vae.to(self.accelerator.device, dtype=torch.float32, non_blocking=True)

    def _unet(self, **kwargs) -> UNet2DConditionModel:
        unet: UNet2DConditionModel = self._spawn(
            UNet2DConditionModel,
            subfolder="unet",
            tap=lambda x: x.requires_grad_(False),
            **kwargs,
        )
        unet.enable_xformers_memory_efficient_attention()
        unet = unet.to(
            self.accelerator.device, dtype=self.params.dtype, non_blocking=True
        )
        return unet
        # .to(memory_format=torch.channels_last)
        # return torch.compile(unet, mode="reduce-overhead", fullgraph=True)

    def _noise_scheduler(self):
        return self._spawn(EulerDiscreteScheduler, subfolder="scheduler")

    @main_process_only
    def _init_trackers(self):
        self.accelerator.init_trackers(
            "dreambooth",
            config=self.params.copy(
                update={
                    "model": self.params.model.copy(
                        update={"name": Model().name, "vae": Model().vae}
                    )
                }
            ).dict(),
        )

    @main_process_only
    def _prepare_to_train(self):
        self._init_trackers()
        if self.params.debug_outputs:
            self.tester.log_images(
                self.instance_class.deterministic_prompt,
                list(map(str, images(self.instance_class.data))),
                title="data",
            )

    @property
    def total_steps(self):
        return self._total_steps * self.params.batch_size

    def exceeded_max_steps(self):
        return (
            self.params.validate_every_epochs is None
            and self.total_steps > self.params.validate_after_steps
        )

    @abstractmethod
    def train(self) -> StableDiffusionPipeline:
        pass

    @main_process_only
    @torch.inference_mode()
    def eval(self, pipeline: StableDiffusionPipeline):
        Evaluator(
            self.accelerator.device, self.params, self.instance_class, pipeline
        ).generate()


def get_mem() -> float:
    return torch.cuda.mem_get_info()[1] / 1e9


def get_params() -> HyperParams:
    params = HyperParams()

    match get_mem():
        case float(n) if n < 16:
            params.gradient_accumulation_steps = 2
        case float(n) if n < 24:
            params.batch_size = 1  # 2
        case float(n) if n < 32:
            params.use_diffusers_unet = False
            params.batch_size = 1  # 3
        case float(n):
            params.use_diffusers_unet = False
            params.batch_size = 3

    match torch.cuda.device_count():
        case int(n) if n > 1:
            params.ti_train_epochs //= n
            params.lora_train_epochs //= n

    match torch.cuda.get_device_capability():
        case (8, _):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
            params.dynamo_backend = "inductor"
            os.environ["ACCELERATE_MIXED_PRECISION"] = "bf16"
        case (7, _):
            params.dynamo_backend = "inductor"
            os.environ["ACCELERATE_MIXED_PRECISION"] = "fp16"
        case _:
            params.dynamo_backend = None
            os.environ["ACCELERATE_MIXED_PRECISION"] = "fp16"

    match os.getenv("ACCELERATE_MIXED_PRECISION"):
        case "bf16":
            params.dtype = torch.bfloat16
            params.model.revision = None
        case "fp16":
            params.dtype = torch.float16
            params.model.revision = "fp16"
        case "fp32":
            params.dtype = torch.float32
            params.model.revision = None

    match os.cpu_count():
        case int(n) if n > 1:
            params.loading_workers = min([n, 32])

    return params


def get_model(
    *,
    klass: Type[BaseTrainer],
    instance_images: Optional[list[bytes]] = None,
    instance_path: Optional[Path] = None,
    params: Optional[HyperParams] = None,
):
    if instance_images:
        instance_path = Path(tempfile.mkdtemp())
        for data in instance_images:
            with open(os.path.join(instance_path, hash_bytes(data)), "wb") as f:
                f.write(data)

    if not instance_path:
        raise RuntimeError("No input data!")

    params = params or get_params()
    return klass(
        instance_class=Class(prompt_=params.token, data=instance_path, type_="token"),
        params=params,
    )
