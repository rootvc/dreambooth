import hashlib
import itertools
import json
import math
import os
import tempfile
from contextlib import contextmanager
from datetime import timedelta
from functools import cached_property, lru_cache
from operator import itemgetter
from pathlib import Path
from typing import Any, Callable, Optional, Type, TypeVar, cast

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
import wandb
from accelerate.logging import get_logger
from accelerate.utils import InitProcessGroupKwargs, release_memory
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    get_scheduler,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.cross_attention import LoRACrossAttnProcessor
from peft import (
    LoraConfig,
    LoraModel,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from PIL import Image
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer

from dreambooth.params import Class, HyperParams, Model
from dreambooth.train.accelerators import BaseAccelerator
from dreambooth.train.eval import Evaluator
from dreambooth.train.shared import (
    compile_model,
    dprint,
    main_process_only,
    partition,
    patch_allowed_pipeline_classes,
)
from dreambooth.train.test import Tester

T = TypeVar("T")


torch.backends.cudnn.benchmark = True
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 64 * 2


class ScoreThresholdExceeded(Exception):
    pass


class PromptDataset(Dataset):
    def __init__(self, prompt: str, n: int):
        self.prompt = prompt
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i: int):
        return {"prompt": self.prompt, "index": i}


class CachedLatentsDataset(Dataset):
    def __init__(
        self,
        accelerator: BaseAccelerator,
        dataset: Dataset,
        params: HyperParams,
        vae: AutoencoderKL,
    ):
        self.dataset = dataset
        self.params = params
        self.vae = vae
        self.accelerator = accelerator
        self._length = len(self.dataset)
        self._cached_latents = {}
        self._warmed = False

    def _compute_latents(self, batch: list[dict[str, torch.FloatTensor]]):
        images = list(
            itertools.chain(
                map(itemgetter("instance_image"), batch),
                map(itemgetter("prior_image"), batch),
            )
        )
        images = torch.stack(images).to(
            self.accelerator.device,
            memory_format=torch.contiguous_format,
            dtype=self.params.dtype,
        )
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        latents = latents.squeeze(0).to("cpu").float().pin_memory()

        input_ids = list(
            itertools.chain(
                map(itemgetter("instance_prompt_ids"), batch),
                map(itemgetter("prior_prompt_ids"), batch),
            )
        )
        input_ids = torch.cat(input_ids, dim=0).to(dtype=torch.long)

        return {"input_ids": input_ids, "latents": latents}

    def __len__(self):
        return self._length // self.params.batch_size

    def __getitem__(self, i: int):
        if not self._warmed and i not in self._cached_latents:
            s = self.params.batch_size
            self._cached_latents[i] = self._compute_latents(
                [self.dataset[idx] for idx in range(s * i, s * (i + 1))]
            )
        return self._cached_latents[i]

    def warm(self):
        for i in tqdm(range(len(self)), disable=not self.accelerator.is_main_process):
            self[i]
        self._warmed = True


class DreamBoothDataset(Dataset):
    def __init__(
        self,
        *,
        instance: Class,
        prior: Class,
        tokenizer: CLIPTokenizer,
        size: int,
        augment: bool = True,
    ):
        self.instance = instance
        self.prior = prior
        self.size = size
        self.tokenizer = tokenizer
        self.augment = augment

        self._length = max(len(self.instance_images), len(self.prior_images))

    @cached_property
    def prior_images(self):
        return list(self.prior.data.iterdir())

    @cached_property
    def instance_images(self):
        return list(self.instance.data.iterdir())

    def image_transforms(self, augment: bool = True):
        t = [
            transforms.ToTensor(),
            transforms.Resize(
                self.size,
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
        ]
        if self.augment and augment:
            t += [
                transforms.RandomCrop(self.size),
                transforms.RandomOrder(
                    [
                        transforms.ColorJitter(brightness=0.2, contrast=0.1),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomAdjustSharpness(2, p=0.5),
                    ]
                ),
            ]
        else:
            t += [
                transforms.CenterCrop(self.size),
            ]
        t += [
            transforms.Normalize([0.5], [0.5]),
        ]
        return transforms.Compose(t)

    def __len__(self):
        return self._length

    def __iter__(self):
        return (self[i] for i in range(self._length))

    def _open_image(self, path: Path):
        img = Image.open(path)
        if img.mode == "RGB":
            return img
        else:
            return img.convert("RGB")

    def _instance_image(self, index):
        path = self.instance_images[index % len(self.instance_images)]
        do_augment, index = divmod(index, len(self.instance_images))
        image = self.image_transforms(do_augment)(self._open_image(path))

        return {
            "instance_image": image,
            "instance_prompt_ids": self.tokenizer(
                self.instance.prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids,
        }

    def _prior_image(self, index):
        path = self.prior_images[index % len(self.prior_images)]
        image = self.image_transforms(False)(self._open_image(path))

        return {
            "prior_image": image,
            "prior_prompt_ids": self.tokenizer(
                self.prior.prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids,
        }

    def __getitem__(self, index):
        return {**self._instance_image(index), **self._prior_image(index)}


def hash_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def hash_image(image: Image.Image) -> str:
    return hash_bytes(image.tobytes())


def to_dtype(new_dtype: torch.dtype):
    @contextmanager
    def f(*args: torch.nn.Module):
        dtypes = [model.dtype for model in args]
        for model in args:
            if model.dtype == new_dtype:
                continue
            dprint(f"Converting {model.__class__.__name__} to {new_dtype}")
            model.to(dtype=new_dtype)
        yield
        for model, dtype in zip(args, dtypes):
            if model.dtype == dtype:
                continue
            dprint(f"Restoring {model.__class__.__name__} to {dtype}")
            model.to(dtype=dtype)

    return f


class Trainer:
    UNET_TARGET_MODULES = ["to_q", "to_v", "query", "value"]
    TEXT_ENCODER_TARGET_MODULES = ["q_proj", "v_proj"]

    accelerator: BaseAccelerator

    def __init__(self, *, instance_class: Class, params: HyperParams):
        self.instance_class = instance_class
        self.params = params

        self.priors_dir = Path(tempfile.mkdtemp())
        self.cache_dir = Path(os.getenv("CACHE_DIR", tempfile.mkdtemp()))
        self.metrics_cache = {}

        try:
            from dreambooth.train.accelerators.colossal import (
                ColossalAccelerator as Accelerator,
            )
        except Exception:
            print("ColossalAI not installed, using default Accelerator")
            from dreambooth.train.accelerators.hf import HFAccelerator as Accelerator

        self.accelerator = Accelerator(
            params=self.params,
            mixed_precision=os.getenv("ACCELERATE_MIXED_PRECISION", "fp16"),
            log_with=["wandb"],
            gradient_accumulation_steps=self.params.gradient_accumulation_steps,
            kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(minutes=5))],
        )
        self.logger = get_logger(__name__)
        self.tester = Tester(self.params, self.accelerator, instance_class)

        self.logger.warning(self.accelerator.state, main_process_only=False)
        self.logger.warning(
            f"Available GPU memory: {get_mem():.2f} GB", main_process_only=True
        )
        self.logger.warning(self.params.dict(), main_process_only=True)

        self._total_steps = 0

    @property
    def total_steps(self):
        return self._total_steps * self.params.batch_size

    def compile(self, model: torch.nn.Module, **kwargs):
        return compile_model(model, backend=self.params.dynamo_backend, **kwargs)

    @lru_cache(maxsize=1)
    def token_id(self, tokenizer: CLIPTokenizer):
        return tokenizer.convert_tokens_to_ids(self.params.token)

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

    def _pipeline(
        self,
        unet: Optional[UNet2DConditionModel] = None,
        text_encoder: Optional[CLIPTextModel] = None,
        vae: Optional[AutoencoderKL] = None,
        tokenizer: Optional[CLIPTokenizer] = None,
        **kwargs,
    ) -> StableDiffusionPipeline:
        with patch_allowed_pipeline_classes():
            pipe = self._spawn(
                DiffusionPipeline,
                safety_checker=None,
                low_cpu_mem_usage=True,
                vae=vae or self._vae().eval(),
                unet=(unet or self._unet(compile=True)).eval(),
                text_encoder=(
                    (text_encoder or self._text_encoder(compile=True)).eval()
                ),
                tokenizer=tokenizer or self._tokenizer(),
                **kwargs,
            )

        if "device_map" not in kwargs:
            pipe.to(self.accelerator.device)
        return pipe

    def _text_encoder(self, compile: bool = False, **kwargs) -> CLIPTextModel:
        te = self._spawn(
            CLIPTextModel,
            subfolder="text_encoder",
            tap=lambda x: x.requires_grad_(False),
            **kwargs,
        )
        if "device_map" not in kwargs:
            te.to(self.accelerator.device, non_blocking=True)
        return self.compile(te, do=compile)

    def _noise_scheduler(self):
        return self._spawn(DDPMScheduler, subfolder="scheduler")

    def _vae(self, compile: bool = True, **kwargs):
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
        # vae.enable_slicing()
        vae.enable_xformers_memory_efficient_attention()
        if "device_map" not in kwargs:
            vae.to(self.accelerator.device, non_blocking=True)
        return self.compile(vae, do=compile)

    def _tokenizer(self) -> CLIPTokenizer:
        return self._spawn(AutoTokenizer, subfolder="tokenizer", use_fast=False)

    def _unet(self, compile: bool = False, **kwargs) -> UNet2DConditionModel:
        unet = self._spawn(
            UNet2DConditionModel,
            subfolder="unet",
            tap=lambda x: x.requires_grad_(False),
            **kwargs,
        )
        if "device_map" not in kwargs:
            unet.to(self.accelerator.device, non_blocking=True)

        if self.params.use_diffusers_unet:
            lora_attn_procs = {}
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
                    hidden_size = list(reversed(unet.config.block_out_channels))[
                        block_id
                    ]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = unet.config.block_out_channels[block_id]
                else:
                    raise RuntimeError(f"Unknown attn processor name: {name}")

                lora_attn_procs[name] = LoRACrossAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    rank=self.params.lora_rank,
                )
                if "device_map" not in kwargs:
                    lora_attn_procs[name].to(self.accelerator.device, non_blocking=True)
            unet.set_attn_processor(lora_attn_procs)
            unet.enable_xformers_memory_efficient_attention()
        else:
            unet.enable_xformers_memory_efficient_attention()
            lora_config = LoraConfig(
                r=self.params.lora_rank,
                lora_alpha=1,
                target_modules=self.UNET_TARGET_MODULES,
                lora_dropout=self.params.lora_dropout,
            )
            unet: UNet2DConditionModel = LoraModel(lora_config, unet)
            if "device_map" not in kwargs:
                unet.to(self.accelerator.device, non_blocking=True)

        return self.compile(unet, do=compile)

    @torch.inference_mode()
    def generate_priors(self, progress_bar: bool = False) -> Class:
        dprint("Generating priors...")

        pipeline = self._pipeline(
            unet=self._unet(compile=True),
            text_encoder=self._text_encoder(compile=True),
            torch_dtype=self.params.dtype,
        )
        pipeline.set_progress_bar_config(disable=not progress_bar)

        prompts = PromptDataset(self.params.prior_prompt, self.params.prior_samples)
        loader: DataLoader = self.accelerator.prepare(
            DataLoader(
                prompts,
                batch_size=self.params.batch_size,
                num_workers=self.params.loading_workers,
            )
        )

        images = (
            image
            for example in pipeline.progress_bar(loader)
            for image in pipeline(example["prompt"]).images
        )
        for image in images:
            hash = hash_image(image)
            image.save(self.priors_dir / f"{hash}.jpg")

        del pipeline
        torch.cuda.empty_cache()

        return Class(prompt_=self.params.prior_prompt, data=self.priors_dir)

    @main_process_only
    @torch.no_grad()
    def _validation(self, pipeline: StableDiffusionPipeline, final: bool = False):
        score = self.tester.test_pipe(
            pipeline, "final_validation" if final else "validation"
        )
        torch.cuda.empty_cache()

        self.metrics_cache.update(score)

        flex = ((self.total_steps - self.params.validate_after_steps) // 100) / 100
        img, txt = score["image_alignment"], score["text_alignment"]
        img_align_trg, txt_align_trg = (
            (
                self.params.final_image_alignment_threshold,
                self.params.final_text_alignment_threshold,
            )
            if final
            else (
                self.params.image_alignment_threshold,
                self.params.text_alignment_threshold,
            )
        )
        img_trg, txt_trg = (img_align_trg - flex, txt_align_trg - flex)
        self.metrics_cache.update(
            {
                "score": img + txt,
                "img_trg": img_trg,
                "txt_trg": txt_trg,
                "score_trg": (img_trg + txt_trg) * 0.9,
            }
        )
        self.logger.warning(
            {"img": img, "txt": txt, "img_trg": img_trg, "txt_trg": txt_trg}
        )

        if (
            ((img + txt) > (img_trg + txt_trg) * 0.95)
            and ((img > img_trg) or (txt > txt_trg))
            and (img > 0.95 * img_trg)
        ):
            raise ScoreThresholdExceeded()

    @torch.no_grad()
    def _do_final_validation(
        self,
        unet: UNet2DConditionModel,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        vae: AutoencoderKL,
        check: bool = False,
    ):
        dprint("Final validation...")
        pipeline = self._create_eval_model(
            unet,
            text_encoder,
            tokenizer,
            vae,
            self.params.lora_alpha,
            self.params.lora_text_alpha,
        )
        if check:
            try:
                self._validation(pipeline, final=True)
            except ScoreThresholdExceeded:
                pass
            else:
                pass
                raise RuntimeError("Final validation failed")
            finally:
                wandb.run.log(
                    self.metrics_cache, step=self._total_steps + 1, commit=True
                )
        return pipeline

    @main_process_only
    @torch.no_grad()
    def _do_validation(
        self,
        unet: UNet2DConditionModel,
        models: dict,
    ):
        unet = self.accelerator.unwrap_model(unet, keep_fp32_wrapper=True)
        text_encoder = self.accelerator.unwrap_model(
            models["text_encoder"], keep_fp32_wrapper=True
        )

        with to_dtype(torch.float)(unet, text_encoder, models["vae"]):
            pipeline = self._pipeline(
                unet=unet.eval(),
                text_encoder=text_encoder.eval(),
                tokenizer=models["tokenizer"],
                vae=models["vae"].eval(),
                torch_dtype=self.params.dtype,
            ).to(self.accelerator.device)
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                pipeline.scheduler.config
            )
            try:
                self._validation(pipeline)
            finally:
                del pipeline
                torch.cuda.empty_cache()

    @torch.inference_mode()
    def _create_eval_model(
        self,
        unet_0: UNet2DConditionModel,
        text_encoder_0: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        vae: AutoencoderKL,
        alpha: float,
        alpha_txt: float,
    ):
        unet_state = get_peft_model_state_dict(
            unet_0, state_dict=self.accelerator.get_state_dict(unet_0)
        )
        unet_config = unet_0.get_peft_config_as_dict(inference=True)
        unet_config["lora_alpha"] = alpha
        unet_config["lora_dropout"] = 0.0

        unet = self._unet(compile=False).requires_grad_(False).eval()
        unet = cast(
            UNet2DConditionModel,
            (
                LoraModel(LoraConfig(**unet_config), unet)
                .to(unet_0.device, non_blocking=True)
                .requires_grad_(False)
            ),
        )
        set_peft_model_state_dict(unet, unet_state)
        unet = unet.eval()

        text_encoder = self._text_encoder(compile=False).requires_grad_(False).eval()
        text_encoder.resize_token_embeddings(len(tokenizer))

        embeds_0 = text_encoder_0.model.get_input_embeddings().weight.data
        embeds: torch.Tensor = text_encoder.get_input_embeddings().weight.data
        token_id = self.token_id(tokenizer)
        embeds[token_id] = embeds_0[token_id]

        text_state = get_peft_model_state_dict(
            text_encoder_0, state_dict=self.accelerator.get_state_dict(text_encoder_0)
        )
        text_config = text_encoder_0.get_peft_config_as_dict(inference=True)
        text_config["lora_alpha"] = alpha_txt
        text_config["lora_dropout"] = 0.0

        text_encoder = cast(
            CLIPTextModel,
            (
                LoraModel(LoraConfig(**text_config), text_encoder)
                .to(text_encoder_0.device, non_blocking=True)
                .requires_grad_(False)
            ),
        )
        set_peft_model_state_dict(text_encoder, text_state)
        text_encoder = text_encoder.eval()

        pipeline = self._pipeline(
            unet=unet.to(dtype=self.params.dtype),
            text_encoder=text_encoder.to(dtype=self.params.dtype),
            tokenizer=tokenizer,
            vae=vae.eval().to(dtype=self.params.dtype),
            torch_dtype=self.params.dtype,
        )
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config
        )
        return pipeline

    @torch.inference_mode()
    def _load_model(self, tokenizer: CLIPTokenizer):
        config = json.loads(
            (self.params.model_output_path / "lora_config.json").read_text()
        )
        state = torch.load(
            self.params.model_output_path / "lora_weights.pt",
            map_location=self.accelerator.device,
        )
        unet_state, text_state = partition(state, lambda kv: "text_encoder_" in kv[0])

        unet = self._unet().eval()
        if self.params.use_diffusers_unet:
            unet.load_attn_procs(self.params.model_output_path / "unet")
        else:
            unet = cast(
                UNet2DConditionModel,
                (
                    LoraModel(LoraConfig(**config["unet_peft"]), unet)
                    .to(unet.device, non_blocking=True)
                    .requires_grad_(False)
                    .eval()
                ),
            )
            set_peft_model_state_dict(unet, unet_state)

        text_encoder = self._text_encoder().eval()
        text_encoder.resize_token_embeddings(len(tokenizer))
        embedding = cast(torch.Tensor, text_encoder.get_input_embeddings().weight.data)
        token_embedding = torch.load(
            self.params.model_output_path / "token_embedding.pt",
            map_location=self.accelerator.device,
        )[self.params.token]
        embedding[self.token_id(tokenizer)] = token_embedding

        text_encoder = LoraModel(LoraConfig(**config["text_peft"]), text_encoder).to(
            self.accelerator.device, non_blocking=True
        )
        set_peft_model_state_dict(
            text_encoder,
            {k.removeprefix("text_encoder_"): v for k, v in text_state.items()},
        )

        vae = self._vae(complile=False).eval()

        pipeline = self._pipeline(
            tokenizer=tokenizer,
            unet=unet,
            text_encoder=text_encoder,
            vae=vae,
        )
        pipeline.unet = unet
        pipeline.text_encoder = text_encoder
        pipeline.vae = vae
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config
        )

        return pipeline

    def _do_epoch(
        self,
        epoch: int,
        unet: UNet2DConditionModel,
        loader: DataLoader,
        optimizer: Optimizer,
        models: dict,
    ):
        unet.train()
        models["text_encoder"].train()

        for batch in loader:
            # Convert images to latent space
            latents = batch["latents"].to(
                self.accelerator.device, dtype=self.params.dtype
            )

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                models["noise_scheduler"].config.num_train_timesteps,
                (bsz,),
                device=latents.device,
            )
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = models["noise_scheduler"].add_noise(
                latents, noise, timesteps
            )

            # Get the text embedding for conditioning
            encoder_hidden_states = models["text_encoder"](
                batch["input_ids"].to(self.accelerator.device)
            )[0]
            # Predict the noise residual
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Get the target for loss depending on the prediction type
            if models["noise_scheduler"].config.prediction_type == "epsilon":
                target = noise
            elif models["noise_scheduler"].config.prediction_type == "v_prediction":
                target = models["noise_scheduler"].get_velocity(
                    latents, noise, timesteps
                )
            else:
                raise ValueError(
                    "Unknown prediction type "
                    + models["noise_scheduler"].config.prediction_type
                )

            # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
            model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
            target, target_prior = torch.chunk(target, 2, dim=0)

            # Compute instance loss
            instance_loss = (
                F.mse_loss(model_pred.float(), target.float(), reduction="none")
                .mean([1, 2, 3])
                .mean()
            )

            # Compute prior loss
            prior_loss = F.mse_loss(
                model_pred_prior.float(), target_prior.float(), reduction="mean"
            )

            # Add the prior loss to the instance loss.
            loss = instance_loss + self.params.prior_loss_weight * prior_loss

            self.accelerator.backward(loss)

            if self.accelerator.sync_gradients:
                params_to_clip = itertools.chain(
                    models["unet_params"],
                    models["text_encoder"].parameters(),
                )
                self.accelerator.clip_grad_norm_(
                    params_to_clip, self.params.max_grad_norm
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            models["lr_scheduler"].step()

            if epoch < self.params.ti_train_epochs:
                with torch.no_grad():
                    idx = torch.arange(len(models["tokenizer"])) != self.token_id(
                        models["tokenizer"]
                    )
                    source = models["input_embeddings"][idx]
                    self.accelerator.unwrap_model(
                        models["text_encoder"]
                    ).get_input_embeddings().weight[idx] = source

            if self.accelerator.sync_gradients:
                self._total_steps += 1

            self.accelerator.log(
                {
                    "ti_lr": models["lr_scheduler"].get_last_lr()[0],
                    "text_lr": models["lr_scheduler"].get_last_lr()[1],
                    "unet_lr": models["lr_scheduler"].get_last_lr()[2],
                    "instance_loss": instance_loss.detach().item(),
                    "prior_loss": prior_loss.detach().item(),
                    **self.metrics_cache,
                },
                step=self.total_steps,
            )
            self.metrics_cache = {}

            if self.exceeded_max_steps():
                break

    def exceeded_max_steps(self):
        return (
            self.params.validate_every_epochs is None
            and self.total_steps > self.params.validate_after_steps
        )

    @property
    def validate_every_epochs(self) -> Optional[int]:
        if not self.params.validate_every_epochs:
            return None
        for milestone, epochs in reversed(self.params.validate_every_epochs.items()):
            if self.total_steps >= milestone:
                return epochs

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
    def _persist(
        self,
        unet: UNet2DConditionModel,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
    ):
        dprint("Saving model...")

        config = {}
        state = {}

        if self.params.use_diffusers_unet:
            unet.save_attn_procs(
                self.params.model_output_path / "unet",
                save_function=self.accelerator.save,
            )
        else:
            unet_state = get_peft_model_state_dict(
                unet, state_dict=self.accelerator.get_state_dict(unet)
            )
            state.update(unet_state)
            config["unet_peft"] = unet.get_peft_config_as_dict(inference=True)

        text_state = {
            f"text_encoder_{k}": v
            for k, v in get_peft_model_state_dict(
                text_encoder,
                state_dict=self.accelerator.get_state_dict(text_encoder),
            ).items()
        }
        state.update(text_state)
        config["text_peft"] = text_encoder.get_peft_config_as_dict(inference=True)

        token_embedding = text_encoder.model.get_input_embeddings().weight.data[
            self.token_id(tokenizer)
        ]
        self.accelerator.save(
            {self.params.token: token_embedding},
            self.params.model_output_path / "token_embedding.pt",
        )

        self.accelerator.save(state, self.params.model_output_path / "lora_weights.pt")
        (self.params.model_output_path / "lora_config.json").write_text(
            json.dumps(config)
        )

        wandb.save(str(self.params.model_output_path / "*"), policy="end")

    def _init_text(
        self, init: Optional[torch.Tensor] = None, compile: bool = False, **kwargs
    ):
        tokenizer = self._tokenizer()
        if not tokenizer.add_tokens(self.params.token):
            raise ValueError(f"Token {self.params.token} already in tokenizer")

        text_encoder = self._text_encoder(compile=compile, **kwargs)
        text_encoder.get_input_embeddings().requires_grad_(True)
        text_encoder.resize_token_embeddings(len(tokenizer))

        embeds: torch.Tensor = text_encoder.get_input_embeddings().weight.data

        if init is None:
            src_token_ids = tokenizer.encode(
                self.params.source_token, add_special_tokens=False
            )
            assert len(src_token_ids) == 1
            init = embeds[src_token_ids[0]]

        tkn_id = self.token_id(tokenizer)
        embeds[tkn_id] = init

        lora_text_config = LoraConfig(
            r=self.params.lora_text_rank,
            lora_alpha=1,
            target_modules=self.TEXT_ENCODER_TARGET_MODULES,
            lora_dropout=self.params.lora_text_dropout,
        )
        text_encoder: CLIPTextModel = LoraModel(lora_text_config, text_encoder).to(
            self.accelerator.device, non_blocking=True
        )

        return (tokenizer, text_encoder)

    def _unet_param_source(self, unet: UNet2DConditionModel):
        if self.params.use_diffusers_unet:
            lora_layers = AttnProcsLayers(unet.attn_processors)
            self.accelerator.register_for_checkpointing(lora_layers)
            return lora_layers
        else:
            return unet

    def _prepare_to_train(self):
        with self.accelerator.init():
            tokenizer, text_encoder = self._init_text(compile=False)
            unet = self._unet(compile=False)

        vae = self._vae(compile=False, torch_dtype=self.params.dtype)

        dataset = DreamBoothDataset(
            instance=self.instance_class,
            prior=self.params.prior_class or self.generate_priors(),
            tokenizer=tokenizer,
            size=self.params.model.resolution,
        )
        dprint("Caching latents...")
        dataset = CachedLatentsDataset(self.accelerator, dataset, self.params, vae)
        dataset.warm()
        loader = DataLoader(
            dataset,
            batch_size=1,
            collate_fn=lambda b: b[0],
            shuffle=True,
            pin_memory=True,
            pin_memory_device=self.accelerator.device,
            num_workers=self.params.loading_workers,
        )

        ti_params = [
            p.requires_grad_(True)
            for p in text_encoder.get_input_embeddings().parameters()
        ]
        unet_param_src = self._unet_param_source(unet)

        params = [
            {
                "lr": self.params.ti_learning_rate,
                "params": ti_params,
            },
            {
                "lr": self.params.text_learning_rate,
                "params": list(set(text_encoder.parameters()) - set(ti_params)),
            },
            {"lr": self.params.learning_rate, "params": unet_param_src.parameters()},
        ]

        steps_per_epoch = math.ceil(
            len(loader)
            * self.params.batch_size
            / self.params.gradient_accumulation_steps
        )
        max_train_steps = self.params.train_epochs * steps_per_epoch

        (loader, unet, text_encoder, vae) = self.accelerator.prepare(
            loader, unet, text_encoder, vae
        )
        if unet_param_src is not unet:
            unet_param_src = self.accelerator.prepare(unet_param_src)

        text_encoder = self.compile(text_encoder)
        vae = self.compile(vae)

        optimizer = self.accelerator.optimizer(
            params,
            betas=self.params.betas,
            weight_decay=self.params.weight_decay,
            eps=self.params.epsilon,
        )

        optimizer = self.accelerator.prepare(optimizer)

        steps_per_epoch = math.ceil(
            len(loader)
            * self.params.batch_size
            / self.params.gradient_accumulation_steps
        )  # may have changed post-accelerate
        epochs = math.ceil(max_train_steps / steps_per_epoch)

        lr_scheduler = get_scheduler(
            self.params.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.params.lr_warmup_steps,
            num_training_steps=max_train_steps,
            num_cycles=self.params.lr_cycles,
        )
        lr_scheduler.lr_lambdas[0] = lambda _: 1

        dprint(f"Training for {epochs} epochs. Max steps: {max_train_steps}.")

        if self.accelerator.is_main_process:
            self._init_trackers()

        self.tester.log_images(
            self.instance_class.deterministic_prompt,
            list(map(str, self.instance_class.data.iterdir())),
            title="data",
        )

        return {
            "epochs": epochs,
            "unet": unet,
            "text_encoder": text_encoder,
            "loader": loader,
            "tokenizer": tokenizer,
            "optimizer": optimizer,
            "vae": vae,
            "lr_scheduler": lr_scheduler,
            "noise_scheduler": self._noise_scheduler(),
            "unet_params": unet_param_src.parameters(),
            "input_embeddings": (
                self.accelerator.unwrap_model(text_encoder)
                .get_input_embeddings()
                .weight.data.clone()
                .to(self.accelerator.device, non_blocking=True)
            ),
        }

    def _train(self, models: dict):
        dprint("Starting training...")
        unet, text_encoder, epochs, optimizer, loader, tokenizer, lr_scheduler = (
            models["unet"],
            models["text_encoder"],
            models["epochs"],
            models["optimizer"],
            models["loader"],
            models["tokenizer"],
            models["lr_scheduler"],
        )
        for epoch in range(epochs):
            if self.accelerator.is_main_process:
                dprint(f"Epoch {epoch + 1}/{epochs} (Step {self.total_steps})")

            pg, blr = optimizer.param_groups, lr_scheduler.base_lrs
            if epoch < self.params.ti_train_epochs:
                pg[0]["lr"] = blr[0] = self.params.ti_learning_rate
                pg[1]["lr"] = blr[1] = 0.0
                pg[2]["lr"] = blr[2] = 0.0
            else:
                pg[0]["lr"] = blr[0] = 0.0
                pg[1]["lr"] = blr[1] = self.params.text_learning_rate
                pg[2]["lr"] = blr[2] = self.params.learning_rate
                if epoch == self.params.ti_train_epochs:
                    dprint(f"Finished TI training at epoch {epoch}.")
                    del models["input_embeddings"]
                    torch.cuda.empty_cache()

            self._do_epoch(epoch, unet, loader, optimizer, models)
            if self.exceeded_max_steps():
                dprint("Max steps exceeded. Stopping training.")
                break

            if (
                self.validate_every_epochs is not None
                and self.total_steps >= self.params.validate_after_steps
                and epoch % self.validate_every_epochs == 0
            ):
                try:
                    self._do_validation(unet, models)
                except ScoreThresholdExceeded:
                    dprint("Score threshold exceeded. Stopping training.")
                    break

        self.accelerator.wait_for_everyone()
        unet = self.accelerator.unwrap_model(unet, keep_fp32_wrapper=False)
        text_encoder = self.accelerator.unwrap_model(
            text_encoder, keep_fp32_wrapper=False
        )
        return tokenizer, unet, text_encoder, models["vae"]

    def train(self):
        self.accelerator.free_memory()
        objs = self._prepare_to_train()
        release_memory()
        tokenizer, unet, text_encoder, vae = self._train(objs)
        self.accelerator.free_memory()
        self.accelerator.wait_for_everyone()
        return self._do_final_validation(unet, text_encoder, tokenizer, vae, check=True)

    @torch.distributed.elastic.multiprocessing.errors.record
    def eval(self, pipeline: StableDiffusionPipeline):
        self.accelerator.wait_for_everyone()
        self.accelerator.free_memory()
        Evaluator(self.accelerator.device, self.params, pipeline).generate()


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
            params.train_epochs //= n

    match torch.cuda.get_device_capability():
        case (8, _):
            torch.backends.cuda.matmul.allow_tf32 = True
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
    return Trainer(
        instance_class=Class(prompt_=params.token, data=instance_path, type_="token"),
        params=params,
    )
