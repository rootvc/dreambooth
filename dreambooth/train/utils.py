import copy
import hashlib
import itertools
import json
import math
import os
import tempfile
from contextlib import contextmanager
from dataclasses import replace
from datetime import timedelta
from functools import cached_property, lru_cache, partial
from operator import itemgetter
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Type, TypeVar, Union, cast

import torch
import torch._dynamo
import torch._dynamo.config
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
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer

from dreambooth.params import Class, HyperParams, Model
from dreambooth.train.accelerators import BaseAccelerator
from dreambooth.train.eval import Evaluator
from dreambooth.train.shared import (
    compile_model,
    dprint,
    main_process_only,
    patch_allowed_pipeline_classes,
)

T = TypeVar("T")


torch.backends.cudnn.benchmark = True
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 64 * 2


class PromptDataset(Dataset):
    def __init__(self, prompt: str, n: int):
        self.prompt = prompt
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i: int):
        return {"prompt": self.prompt, "index": i}


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

    def loader_collate_fn(self, examples: list[dict]):
        input_ids = list(
            itertools.chain(
                map(itemgetter("instance_prompt_ids"), examples),
                map(itemgetter("prior_prompt_ids"), examples),
            )
        )
        pixel_values = list(
            itertools.chain(
                map(itemgetter("instance_image"), examples),
                map(itemgetter("prior_image"), examples),
            )
        )

        input_ids = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        return {
            "input_ids": input_ids,
            "pixel_values": torch.stack(pixel_values)
            .to(memory_format=torch.contiguous_format)
            .float(),
        }

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
                padding="do_not_pad",
                max_length=self.tokenizer.model_max_length,
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
                padding="do_not_pad",
                max_length=self.tokenizer.model_max_length,
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
            print(f"Converting {model.__class__.__name__} to {new_dtype}")
            model.to(dtype=new_dtype)
        yield
        for model, dtype in zip(args, dtypes):
            print(f"Restoring {model.__class__.__name__} to {dtype}")
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
        self.logger.warning(self.accelerator.state, main_process_only=False)
        self.logger.warning(
            f"Available GPU memory: {get_mem():.2f} GB", main_process_only=True
        )
        self.logger.warning(self.params.dict(), main_process_only=True)

        self._total_steps = 0

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
                local_files_only=True,
                vae=vae or self._vae().eval(),
                unet=unet.eval() or self._unet(compile=True).eval(),
                text_encoder=(
                    text_encoder.eval() or self._text_encoder(compile=True).eval()
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
            te.to(self.accelerator.device)
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
        vae.enable_slicing()
        if "device_map" not in kwargs:
            vae.to(self.accelerator.device)
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
            unet.to(self.accelerator.device)

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
                    lora_attn_procs[name].to(self.accelerator.device)
            unet.set_attn_processor(lora_attn_procs)
            unet.enable_xformers_memory_efficient_attention()
        else:
            unet.enable_xformers_memory_efficient_attention()
            lora_config = LoraConfig(
                r=self.params.lora_rank,
                lora_alpha=self.params.lora_alpha,
                target_modules=self.UNET_TARGET_MODULES,
                lora_dropout=self.params.lora_dropout,
            )
            unet: UNet2DConditionModel = LoraModel(lora_config, unet)
            if "device_map" not in kwargs:
                unet.to(self.accelerator.device)

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
    def _log_images(
        self,
        prompts: Union[str, list[str]],
        images: Iterable,
        title: str = "validation",
    ):
        images = list(images)
        if not isinstance(prompts, list):
            prompts = [prompts] * len(images)

        self.accelerator.wandb_tracker.log(
            {
                title: [
                    wandb.Image(image, caption=f"{i}: {prompts[i]}")
                    for i, image in enumerate(images)
                ]
            }
        )

    @main_process_only
    @torch.no_grad()
    def _validation(
        self, pipeline: StableDiffusionPipeline, title: str = "validation"
    ) -> list:
        prompt = (
            self.instance_class.deterministic_prompt
            + ", "
            + self.params.validation_prompt_suffix
        )
        generator = torch.Generator(device=self.accelerator.device)

        images = pipeline(
            prompt,
            negative_prompt=self.params.negative_prompt,
            num_inference_steps=self.params.validation_steps,
            num_images_per_prompt=self.params.validation_samples,
            generator=generator,
        ).images

        self._log_images([prompt] * self.params.validation_samples, images, title=title)
        return images

    @torch.no_grad()
    def _do_final_validation(
        self,
        unet: UNet2DConditionModel,
        text_encoder: CLIPTextModel,
        models: dict,
    ):
        pipeline = self._create_eval_model(unet, text_encoder, models["tokenizer"])
        self._validation(pipeline, title="final_validation")
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
                unet=unet,
                text_encoder=text_encoder,
                tokenizer=models["tokenizer"],
                vae=models["vae"],
                torch_dtype=self.params.dtype,
            )
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                pipeline.scheduler.config
            )

            pipeline.set_progress_bar_config(disable=True)
            self._validation(pipeline)

        del pipeline
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def _create_eval_model(
        self,
        unet_: UNet2DConditionModel,
        text_encoder_: CLIPTextModel,
        tokenizer: CLIPTokenizer,
    ):
        unet = self._unet(torch_dtype=self.params.dtype).eval()
        if self.params.use_diffusers_unet:
            unet.load_attn_procs(self.params.model_output_path / "unet")
        else:
            unet = cast(
                UNet2DConditionModel,
                (
                    LoraModel(replace(unet_.peft_config, inference_mode=True), unet)
                    .to(unet.device, dtype=self.params.dtype)
                    .requires_grad_(False)
                    .eval()
                ),
            )
            set_peft_model_state_dict(unet, get_peft_model_state_dict(unet_))

        unet_.to_empty(device="cpu")
        del unet_

        text_encoder = self._text_encoder(torch_dtype=self.params.dtype).eval()
        text_encoder.resize_token_embeddings(len(tokenizer))
        embedding = cast(torch.Tensor, text_encoder.get_input_embeddings().weight.data)
        embedding_ = cast(
            torch.Tensor, text_encoder_.get_input_embeddings().weight.data
        )
        embedding[self.token_id(tokenizer)] = copy.deepcopy(
            embedding_[self.token_id(tokenizer)]
        ).to(embedding.device, dtype=embedding.dtype)
        text_encoder = cast(
            CLIPTextModel,
            (
                LoraModel(
                    replace(text_encoder_.peft_config, inference_mode=True),
                    text_encoder,
                )
                .to(text_encoder.device, dtype=self.params.dtype)
                .requires_grad_(False)
                .eval()
            ),
        )
        set_peft_model_state_dict(
            text_encoder,
            get_peft_model_state_dict(text_encoder_),
        )

        text_encoder_.to_empty(device="cpu")
        del text_encoder_

        vae = self._vae(torch_dtype=self.params.dtype).eval()

        pipeline = self._pipeline(
            unet=unet.to(dtype=self.params.dtype),
            text_encoder=text_encoder.to(dtype=self.params.dtype),
            tokenizer=tokenizer,
            vae=vae.to(dtype=self.params.dtype),
            torch_dtype=self.params.dtype,
        )
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config
        )

        return pipeline

    def _do_epoch(
        self,
        epoch: int,
        unet: UNet2DConditionModel,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        models: dict,
    ):
        unet.train()
        models["text_encoder"].train()

        for batch in loader:
            # Convert images to latent space
            latents = (
                models["vae"]
                .encode(batch["pixel_values"].to(dtype=self.params.dtype))
                .latent_dist.sample()
            )
            latents = latents * models["vae"].config.scaling_factor

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
                batch["input_ids"].to(self.accelerator.device, dtype=torch.long)
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
            loss = (
                F.mse_loss(model_pred.float(), target.float(), reduction="none")
                .mean([1, 2, 3])
                .mean()
            )

            # Compute prior loss
            prior_loss = F.mse_loss(
                model_pred_prior.float(), target_prior.float(), reduction="mean"
            )

            # Add the prior loss to the instance loss.
            loss = loss + self.params.prior_loss_weight * prior_loss

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
            if not self.accelerator.optimizer_step_was_skipped:
                models["lr_scheduler"].step()
            optimizer.zero_grad(set_to_none=True)

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
                self._total_steps += self.params.batch_size

            self.accelerator.log(
                {
                    "loss": loss.detach().item(),
                    "ti_lr": models["lr_scheduler"].get_last_lr()[0],
                    "text_lr": models["lr_scheduler"].get_last_lr()[1],
                    "unet_lr": models["lr_scheduler"].get_last_lr()[2],
                },
                step=self._total_steps,
            )

            if self.exceeded_max_steps():
                break

    def exceeded_max_steps(self):
        return (
            self.params.validate_every_epochs is None
            and self._total_steps > self.params.validate_after_steps
        )

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

        token_embedding = text_encoder.get_input_embeddings().weight.data[
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
            lora_alpha=self.params.lora_text_alpha,
            target_modules=self.TEXT_ENCODER_TARGET_MODULES,
            lora_dropout=self.params.lora_text_dropout,
        )
        text_encoder: CLIPTextModel = LoraModel(lora_text_config, text_encoder).to(
            self.accelerator.device
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
            tokenizer, text_encoder = self._init_text()
            unet = self._unet()

        unet = self.compile(unet, do=False)
        text_encoder = self.compile(text_encoder, do=False)

        dataset = DreamBoothDataset(
            instance=self.instance_class,
            prior=self.params.prior_class or self.generate_priors(),
            tokenizer=tokenizer,
            size=self.params.model.resolution,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.params.batch_size,
            shuffle=True,
            pin_memory=True,
            pin_memory_device=self.accelerator.device,
            collate_fn=dataset.loader_collate_fn,
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

        (loader, unet, text_encoder) = self.accelerator.prepare(
            loader, unet, text_encoder
        )
        if unet_param_src is not unet:
            unet_param_src = self.accelerator.prepare(unet_param_src)

        optimizer = self.accelerator.optimizer(
            params,
            betas=self.params.betas,
            weight_decay=self.params.weight_decay,
            eps=self.params.epsilon,
        )

        optimizer = self.accelerator.prepare(optimizer, device_placement=[False])

        steps_per_epoch = math.ceil(
            len(loader)
            * self.params.batch_size
            / self.params.gradient_accumulation_steps
        )  # may have changed post-accelerate
        epochs = math.ceil(max_train_steps / steps_per_epoch)

        def linear_with_warmup(
            skip_steps: int, warmup_steps: int, training_steps: int, current_step: int
        ):
            if current_step < skip_steps:
                return 0.0

            current_step -= skip_steps
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0,
                float(training_steps - current_step)
                / float(max(1, training_steps - warmup_steps)),
            )

        lr_scheduler = LambdaLR(
            optimizer,
            [
                lambda step: 1.0,
                partial(
                    linear_with_warmup,
                    self.params.ti_train_epochs * steps_per_epoch,
                    (
                        self.params.lr_warmup_steps
                        * self.params.gradient_accumulation_steps
                    ),
                    max_train_steps * self.params.gradient_accumulation_steps,
                ),
                partial(
                    linear_with_warmup,
                    self.params.ti_train_epochs * steps_per_epoch,
                    (
                        self.params.lr_warmup_steps
                        * self.params.gradient_accumulation_steps
                    ),
                    max_train_steps * self.params.gradient_accumulation_steps,
                ),
            ],
        )
        lr_scheduler = self.accelerator.prepare(lr_scheduler)

        dprint(f"Training for {epochs} epochs. Max steps: {max_train_steps}.")

        if self.accelerator.is_main_process:
            self._init_trackers()

        self._log_images(
            self.instance_class.deterministic_prompt,
            map(str, self.instance_class.data.iterdir()),
            title="data",
        )

        return {
            "epochs": epochs,
            "unet": unet,
            "text_encoder": text_encoder,
            "loader": loader,
            "tokenizer": tokenizer,
            "optimizer": optimizer,
            "vae": self._vae(torch_dtype=self.params.dtype),
            "noise_scheduler": self._noise_scheduler(),
            "lr_scheduler": lr_scheduler,
            "unet_params": unet_param_src.parameters(),
            "input_embeddings": (
                self.accelerator.unwrap_model(text_encoder)
                .get_input_embeddings()
                .weight.data.clone()
                .to(self.accelerator.device)
            ),
        }

    def _train(self, models: dict):
        dprint("Starting training...")
        unet, text_encoder, epochs, optimizer, loader, tokenizer = (
            models["unet"],
            models["text_encoder"],
            models["epochs"],
            models["optimizer"],
            models["loader"],
            models["tokenizer"],
        )
        for epoch in range(epochs):
            self.logger.warning(
                f"Epoch {epoch + 1}/{epochs} (Step {self._total_steps})",
                main_process_only=True,
            )

            if epoch < self.params.ti_train_epochs:
                optimizer.param_groups[0]["lr"] = self.params.ti_learning_rate
                optimizer.param_groups[1]["lr"] = 0.0
                optimizer.param_groups[2]["lr"] = 0.0
            else:
                optimizer.param_groups[0]["lr"] = 0.0
                optimizer.param_groups[1]["lr"] = self.params.text_learning_rate
                optimizer.param_groups[0]["lr"] = self.params.learning_rate
                if epoch == self.params.ti_train_epochs:
                    dprint(f"Finished TI training at epoch {epoch}.")
                    del models["input_embeddings"]
                    torch.cuda.empty_cache()

            self._do_epoch(epoch, unet, loader, optimizer, models)
            if self.exceeded_max_steps():
                self.accelerator.wait_for_everyone()
                self._do_validation(unet, models)
                break

            if (
                self.params.validate_every_epochs is not None
                and self._total_steps >= self.params.validate_after_steps
                and epoch % self.params.validate_every_epochs == 0
            ):
                self.accelerator.wait_for_everyone()
                self._do_validation(unet, models)

        self.accelerator.wait_for_everyone()
        unet = self.accelerator.unwrap_model(unet, keep_fp32_wrapper=False)
        text_encoder = self.accelerator.unwrap_model(
            text_encoder, keep_fp32_wrapper=False
        )
        self._persist(unet, text_encoder, tokenizer)
        return unet, text_encoder, models

    def train(self):
        self.accelerator.free_memory()
        objs = self._prepare_to_train()
        release_memory()
        unet, text_encoder, models = self._train(objs)
        self.accelerator.free_memory()
        return self._do_final_validation(unet, text_encoder, models)

    @torch.distributed.elastic.multiprocessing.errors.record
    def eval(self, pipeline: StableDiffusionPipeline):
        self.accelerator.wait_for_everyone()
        self.accelerator.free_memory()

        pipeline.unet = self.compile(pipeline.unet, do=False)
        pipeline.text_encoder = self.compile(pipeline.text_encoder, do=False)
        Evaluator(self.accelerator, self.params, pipeline).generate()


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
            params.batch_size = 1  # int(n / 8)

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
