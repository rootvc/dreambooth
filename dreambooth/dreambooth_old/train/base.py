import itertools
import math
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
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers.utils.logging
from accelerate.logging import get_logger
from accelerate.utils import InitProcessGroupKwargs, release_memory
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    SchedulerMixin,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
    get_scheduler,
)
from diffusers.models.attention_processor import LoRAAttnProcessor2_0
from torch.utils.data import DataLoader
from transformers import CLIPTextModel

from dreambooth_old.params import Class, HyperParams, Model
from dreambooth_old.train.accelerators import BaseAccelerator
from dreambooth_old.train.datasets import (
    CachedLatentsDataset,
    DreamBoothDataset,
    PromptDataset,
)
from dreambooth_old.train.eval import Evaluator
from dreambooth_old.train.sdxl.utils import get_variance_type
from dreambooth_old.train.shared import (
    dprint,
    hash_bytes,
    hash_image,
    images,
    main_process_only,
    unpack_collate,
)
from dreambooth_old.train.test import Tester

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
            from dreambooth_old.train.accelerators.colossal import (
                ColossalAccelerator as Accelerator,
            )
        except Exception:
            print("ColossalAI not installed, using default Accelerator")
            from dreambooth_old.train.accelerators.hf import (
                HFAccelerator as Accelerator,
            )

        return Accelerator(
            params=self.params,
            mixed_precision=os.getenv("ACCELERATE_MIXED_PRECISION", "bf16"),
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
            **kwargs,
        )
        tap(instance)
        return instance

    def _vae(self, **kwargs):
        if self.params.model.vae:
            vae = AutoencoderKL.from_pretrained(
                self.params.model.vae,
                **kwargs,
            )
        else:
            vae = self._spawn(
                AutoencoderKL,
                subfolder="vae",
                **kwargs,
            )

        vae.requires_grad_(False)
        # vae.enable_xformers_memory_efficient_attention()
        return vae.to(self.accelerator.device, dtype=torch.float32, non_blocking=True)

    def _unet(self, **kwargs) -> UNet2DConditionModel:
        unet: UNet2DConditionModel = self._spawn(
            UNet2DConditionModel,
            subfolder="unet",
            tap=lambda x: x.requires_grad_(False),
            **kwargs,
        )
        # unet.enable_xformers_memory_efficient_attention()
        unet = unet.to(
            self.accelerator.device, dtype=self.params.dtype, non_blocking=True
        )
        unet = unet.to(memory_format=torch.channels_last)
        return torch.compile(unet, mode="reduce-overhead", fullgraph=True)

    def _unet_for_lora(self, **kwargs):
        unet = self._unet(**kwargs)
        unet_lora_attn_procs = {}
        unet_lora_parameters = []
        for name in unet.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            else:
                raise ValueError(f"unexpected attn processor name: {name}")

            module = LoRAAttnProcessor2_0(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=self.params.lora_rank,
            )
            unet_lora_attn_procs[name] = module
            unet_lora_parameters.extend(module.parameters())

        unet.set_attn_processor(unet_lora_attn_procs)
        return unet, unet_lora_parameters

    @abstractmethod
    def _pipeline(self, klass: type, *args, **kwargs) -> StableDiffusionPipeline:
        ...

    @torch.inference_mode()
    def _generate_priors_with(
        self, klass: type, gen_kwargs: dict = {}, **kwargs
    ) -> Class:
        pipeline = self._pipeline(klass=klass, **kwargs)

        prompts = PromptDataset(self.params.prior_prompt, self.params.prior_samples)
        loader = DataLoader(prompts, batch_size=self.params.batch_size)
        images = (
            pipeline(
                prompt=batch["prompt"],
                negative_prompt=[self.params.negative_prompt] * len(batch["prompt"]),
                **gen_kwargs,
            ).images
            for batch in loader
        )
        for image in itertools.chain.from_iterable(images):
            hash = hash_image(image)
            image.save(self.priors_dir / f"{hash}.jpg")

        pipeline = release_memory(pipeline)
        return Class(prompt_=self.params.prior_prompt, data=self.priors_dir)

    @abstractmethod
    def generate_priors(self) -> Class:
        ...

    @abstractmethod
    def _load_models(self) -> tuple:
        ...

    def _prepare_models(self):
        (
            unet,
            text_encoders,
            vae,
            tokenizers,
            noise_scheduler,
            optimizer,
            params,
        ) = self._load_models()
        dataset = DreamBoothDataset(
            instance=self.instance_class,
            prior=self.params.prior_class or self.generate_priors(),
            tokenizers=tokenizers,
            params=self.params,
            vae_scale_factor=vae.config.scaling_factor,
        )

        dprint("Caching latents...")
        dataset = CachedLatentsDataset(self.accelerator, dataset, self.params, vae)
        warmed = dataset.warm()

        dprint("Loading data...")
        loader = DataLoader(
            warmed,
            batch_size=1,
            collate_fn=unpack_collate,
            shuffle=True,
            num_workers=1,
        )

        epochs = self.params.lora_train_epochs
        steps_per_epoch = math.ceil(
            len(loader)
            * self.params.batch_size
            / self.params.gradient_accumulation_steps
        )
        max_train_steps = epochs * steps_per_epoch

        lr_scheduler = get_scheduler(
            self.params.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.params.lr_warmup_steps
            * self.accelerator.num_processes,
            num_training_steps=max_train_steps * self.accelerator.num_processes,
            num_cycles=self.params.lr_cycles,
            power=2.0,
        )

        (
            unet,
            *text_encoders,
            optimizer,
            loader,
            lr_scheduler,
        ) = self.accelerator.prepare(
            unet,
            *text_encoders,
            optimizer,
            loader,
            lr_scheduler,
        )

        steps_per_epoch = math.ceil(
            len(loader)
            * self.params.batch_size
            / self.params.gradient_accumulation_steps
        )  # may have changed post-accelerate
        epochs = math.ceil(max_train_steps / steps_per_epoch)

        return (
            unet,
            text_encoders,
            vae,
            tokenizers,
            noise_scheduler,
            optimizer,
            loader,
            lr_scheduler,
            epochs,
            params,
        )

    @abstractmethod
    def _unet_epoch_args(
        self, batch: dict, bsz: int, text_encoders: list
    ) -> tuple[list, dict]:
        ...

    def _do_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        noise_scheduler: SchedulerMixin,
        unet: UNet2DConditionModel,
        text_encoders: list[CLIPTextModel],
        scaling_factor: float,
        lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
        vae: AutoencoderKL,
        tokenizers,
        params,
    ):
        unet.train()
        for text_encoder in text_encoders:
            text_encoder.train()

        for batch in loader:
            latent_dist = batch["latent_dist"].to(self.accelerator.device)
            latents = latent_dist.sample()
            latents = latents * scaling_factor
            latents = latents.squeeze(0).float()

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
            )
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(
                latents,
                noise,  # + self.params.input_perterbation * torch.randn_like(latents),
                timesteps,
            )

            args, kwargs = self._unet_epoch_args(batch, bsz, text_encoders)
            model_pred = unet(noisy_latents, timesteps, *args, **kwargs).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                )

            # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
            model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
            target, target_prior = torch.chunk(target, 2, dim=0)

            instance_loss = F.mse_loss(
                model_pred.float(), target.float(), reduction="mean"
            )
            prior_loss = F.mse_loss(
                model_pred_prior.float(), target_prior.float(), reduction="mean"
            )
            loss = instance_loss + self.params.prior_loss_weight * prior_loss
            self.accelerator.backward(loss)

            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(params, self.params.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if self.accelerator.sync_gradients:
                self._total_steps += 1
            metrics = {
                "instance_loss": instance_loss.detach().item(),
                "prior_loss": prior_loss.detach().item(),
                "steps": self.total_steps,
                "lr": lr_scheduler.get_last_lr()[0],
                "text_lr": lr_scheduler.get_last_lr()[1],
                **self.metrics_cache,
            }
            self.accelerator.log(metrics, step=self.total_steps)
            self.metrics_cache = {}

            if self.exceeded_max_steps():
                break

    def _noise_scheduler(self):
        return self._spawn(DDPMScheduler, subfolder="scheduler")

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
    def _unwrap_pipe_args(
        self, unet, text_encoders, tokenizers, vae
    ) -> tuple[list, dict]:
        ...

    def train(self):
        self._prepare_to_train()
        (
            unet,
            text_encoders,
            vae,
            tokenizers,
            noise_scheduler,
            optimizer,
            loader,
            lr_scheduler,
            epochs,
            params,
        ) = self._prepare_models()

        release_memory()

        dprint("Starting training...")
        for epoch in range(epochs):
            if self.accelerator.is_main_process:
                dprint(f"Epoch {epoch + 1}/{epochs} (Step {self.total_steps})")

            self._do_epoch(
                loader,
                optimizer,
                noise_scheduler,
                unet,
                text_encoders,
                vae.config.scaling_factor,
                lr_scheduler,
                vae,
                tokenizers,
                params,
            )

            if self.exceeded_max_steps():
                dprint("Max steps exceeded. Stopping training.")
                break

        self.accelerator.wait_for_everyone()

        args, kwargs = self._unwrap_pipe_args(unet, text_encoders, tokenizers, vae)
        pipe = self._pipeline(*args, **kwargs, torch_dtype=self.params.dtype)
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config,
            disable_corrector=[0],
            use_karras_sigmas=True,
            timestep_spacing="trailing",
            **get_variance_type(pipe.scheduler),
        )
        return pipe.to(self.accelerator.device)

    @main_process_only
    @torch.inference_mode()
    def eval(self, pipeline: StableDiffusionPipeline):
        Evaluator(
            self.accelerator.device, self.params, self.instance_class, pipeline
        ).generate()

    def end_training(self):
        self.accelerator.end_training()
        self.accelerator.free_memory()


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
        case (8, _) | (9, _):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
            torch._dynamo.config.suppress_errors = True
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
