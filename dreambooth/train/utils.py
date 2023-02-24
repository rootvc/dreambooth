import functools
import hashlib
import itertools
import json
import math
import os
import tempfile
from functools import cached_property
from operator import itemgetter
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Type, TypeVar

import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.tracking import WandBTracker
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from peft import (
    LoraConfig,
    LoraModel,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from PIL import Image
from torch._dynamo import disable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoTokenizer, CLIPTextModel
from transformers.modeling_utils import PreTrainedModel

from dreambooth.params import Class, HyperParams

T = TypeVar("T")


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
        tokenizer: PreTrainedModel,
        size: int,
        augment: bool = True,
    ):
        self.instance = instance
        self.prior = prior
        self.size = size
        self.tokenizer = tokenizer
        self.augment = augment

        self._length = max(len(self.instance_images), len(self.prior_images))

    @staticmethod
    def loader_collate_fn(examples: list[dict]):
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
        return {
            "input_ids": torch.cat(input_ids, dim=0),
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
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
        ]
        if self.augment and augment:
            t += [
                transforms.RandomCrop(self.size),
                transforms.ColorJitter(0.1, 0.1),
                transforms.RandomErasing(p=0.5),
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


def _main_process_only(f):
    @functools.wraps(f)
    def wrapper(self: "Trainer", *args, **kwargs):
        if not self.accelerator.is_main_process:
            return
        return f(self, *args, **kwargs)

    return wrapper


def partition(
    d: dict[str, T], pred: Callable[[tuple[str, T]], bool]
) -> tuple[dict[str, T], dict[str, T]]:
    t1, t2 = itertools.tee(d.items())
    return tuple(map(dict, [itertools.filterfalse(pred, t1), filter(pred, t2)]))


class Trainer:
    UNET_TARGET_MODULES = ["to_q", "to_v", "query", "value"]
    TEXT_ENCODER_TARGET_MODULES = ["q_proj", "v_proj"]

    def __init__(self, *, instance_class: Class, params: HyperParams):
        self.instance_class = instance_class
        self.params = params

        self.priors_dir = Path(tempfile.mkdtemp())
        self.output_dir = Path(tempfile.mkdtemp())

        self.accelerator = Accelerator(
            mixed_precision=os.getenv("ACCELERATE_MIXED_PRECISION", "fp16"),
            log_with=["wandb"],
            gradient_accumulation_steps=self.params.gradient_accumulation_steps,
        )
        self.logger = get_logger(__name__)
        self.logger.warning(self.accelerator.state, main_process_only=True)
        self.logger.warning(self.params.dict(), main_process_only=True)

        self._total_steps = 0

    @_main_process_only
    def _print(self, *args, **kwargs):
        print(*args, **kwargs)

    def _spawn(
        self, klass: Type[T], tap: Callable[[T], Any] = lambda x: x, **kwargs
    ) -> T:
        instance = klass.from_pretrained(
            self.params.model.name,
            revision=self.params.model.revision,
            torch_dtype=self.params.dtype,
            **kwargs,
        )
        tap(instance)
        return instance

    def _pipeline(
        self,
        unet: Optional[UNet2DConditionModel] = None,
        text_encoder: Optional[CLIPTextModel] = None,
        half: bool = False,
        **kwargs,
    ):
        pipe = self._spawn(
            DiffusionPipeline,
            safety_checker=None,
            low_cpu_mem_usage=True,
            local_files_only=True,
            unet=self._unet().eval(),
            text_encoder=self._text_encoder().eval(),
            vae=self._vae().eval(),
            tokenizer=self._tokenizer(),
            **kwargs,
        )
        if unet:
            pipe.unet = unet.eval()
        if text_encoder:
            pipe.text_encoder = text_encoder.eval()
        if half:
            pipe.unet.half()
            pipe.text_encoder.half()
        return pipe.to(self.accelerator.device)

    def _text_encoder(self):
        return self._spawn(
            CLIPTextModel,
            subfolder="text_encoder",
            tap=lambda x: x.requires_grad_(False),
        ).to(self.accelerator.device, dtype=self.params.dtype)

    def _noise_scheduler(self):
        return self._spawn(DDPMScheduler, subfolder="scheduler")

    def _vae(self):
        if self.params.model.vae:
            vae = AutoencoderKL.from_pretrained(
                self.params.model.vae,
            )
        else:
            vae = self._spawn(
                AutoencoderKL,
                subfolder="vae",
            )

        vae.requires_grad_(False)
        return vae.to(self.accelerator.device, dtype=self.params.dtype)

    def _tokenizer(self):
        return self._spawn(
            AutoTokenizer,
            subfolder="tokenizer",
            use_fast=False,
        )

    def _unet(self):
        unet = self._spawn(
            UNet2DConditionModel,
            subfolder="unet",
            tap=lambda x: x.requires_grad_(False),
        ).to(self.accelerator.device, dtype=self.params.dtype)
        try:
            unet.enable_xformers_memory_efficient_attention()
        except Exception as e:
            self._print("Cannot enable xformers memory efficient attention")
            self._print(e)
        return unet

    @torch.inference_mode()
    def generate_priors(self, progress_bar: bool = False) -> Class:
        self._print("Generating priors...")

        pipeline = self._pipeline(
            unet=self._unet(),
            text_encoder=self._text_encoder(),
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

        return Class(prompt=self.params.prior_prompt, data=self.priors_dir)

    @_main_process_only
    def _log_images(self, prompt: str, images: Iterable, title: str = "validation"):
        for tracker in self.accelerator.trackers:
            if not isinstance(tracker, WandBTracker):
                continue
            tracker.log(
                {
                    title: [
                        wandb.Image(image, caption=f"{i}: {prompt}")
                        for i, image in enumerate(images)
                    ]
                }
            )

    @torch.no_grad()
    def _validation(self, pipeline: DiffusionPipeline) -> list:
        prompt = self.instance_class.prompt + " " + self.params.validation_prompt_suffix
        generator = torch.Generator(device=self.accelerator.device)
        images = [
            pipeline(
                prompt,
                num_inference_steps=self.params.validation_steps,
                generator=generator,
            ).images[0]
            for _ in range(self.params.validation_samples)
        ]

        self._log_images(prompt, images, title="validation")
        return images

    @_main_process_only
    @disable
    def _do_validation(
        self,
        unet: UNet2DConditionModel,
        models: dict,
    ):
        unet = self.accelerator.unwrap_model(unet, keep_fp32_wrapper=True)
        text_encoder = self.accelerator.unwrap_model(
            models["text_encoder"], keep_fp32_wrapper=True
        )
        self._persist(unet, text_encoder)
        return self._do_final_validation()
        pipeline = self._pipeline(unet=unet, text_encoder=text_encoder)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config
        )
        pipeline.set_progress_bar_config(disable=True)

        self._validation(pipeline)

        del pipeline
        torch.cuda.empty_cache()

    @_main_process_only
    @torch.inference_mode()
    def _do_final_validation(self):
        pipeline = self._pipeline(half=True)
        config = json.loads((self.output_dir / "lora_config.json").read_text())
        state = torch.load(self.output_dir / "lora_weights.pt")

        self._print("Loaded config with keys: ", config.keys())

        unet_state, text_state = partition(state, lambda kv: "text_encoder_" in kv[0])

        pipeline.unet = LoraModel(LoraConfig(**config["unet_peft"]), pipeline.unet)
        set_peft_model_state_dict(pipeline.unet, unet_state)

        if self.params.train_text_encoder:
            pipeline.text_encoder = LoraModel(
                LoraModel(LoraConfig(**config["text_peft"]), pipeline.unet),
                pipeline.text_encoder,
            )
            set_peft_model_state_dict(
                pipeline.text_encoder,
                {k.removeprefix("text_encoder_"): v for k, v in text_state.items()},
            )

        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config
        )
        return self._validation(pipeline)

    def _do_epoch(
        self,
        unet: UNet2DConditionModel,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        models: dict,
    ):
        unet.train()
        if self.params.train_text_encoder:
            models["text_encoder"].train()

        for batch in loader:
            with self.accelerator.accumulate(unet):
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
                encoder_hidden_states = models["text_encoder"](batch["input_ids"])[0]
                # Predict the noise residual
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample

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
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Compute prior loss
                prior_loss = F.mse_loss(
                    model_pred_prior.float(), target_prior.float(), reduction="mean"
                )

                # Add the prior loss to the instance loss.
                loss = loss + self.params.prior_loss_weight * prior_loss

                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(
                            unet.parameters(), models["text_encoder"].parameters()
                        )
                        if self.params.train_text_encoder
                        else unet.parameters()
                    )
                    self.accelerator.clip_grad_norm_(
                        params_to_clip, self.params.max_grad_norm
                    )
                optimizer.step()
                models["lr_scheduler"].step()
                optimizer.zero_grad(set_to_none=True)

            if self.accelerator.sync_gradients:
                self._total_steps += 1

            self.accelerator.log(
                {
                    "loss": loss.detach().item(),
                    "lr": models["lr_scheduler"].get_last_lr()[0],
                },
                step=self._total_steps,
            )

    @_main_process_only
    def _init_trackers(self):
        self.accelerator.init_trackers("dreambooth", config=self.params.dict())

    @_main_process_only
    def _persist(self, unet: UNet2DConditionModel, text_encoder: CLIPTextModel):
        config = {}
        state = {}

        unet_state = get_peft_model_state_dict(
            unet, state_dict=self.accelerator.get_state_dict(unet)
        )
        state.update(unet_state)
        config["unet_peft"] = unet.get_peft_config_as_dict(inference=True)

        if self.params.train_text_encoder:
            text_state = {
                f"text_encoder_{k}": v
                for k, v in get_peft_model_state_dict(
                    text_encoder,
                    state_dict=self.accelerator.get_state_dict(text_encoder),
                ).items()
            }
            state.update(text_state)
            config["text_peft"] = text_encoder.get_peft_config_as_dict(inference=True)

            self.accelerator.save(state, self.output_dir / "lora_weights.pt")
            (self.output_dir / "lora_config.json").write_text(json.dumps(config))

            self._print("Persisted config with keys: ", config.keys())

    def train(self):
        lora_config = LoraConfig(
            r=self.params.lora_rank,
            lora_alpha=self.params.lora_alpha,
            target_modules=self.UNET_TARGET_MODULES,
            lora_dropout=self.params.lora_dropout,
        )
        unet = LoraModel(lora_config, self._unet())

        if self.params.train_text_encoder:
            lora_text_config = LoraConfig(
                r=self.params.lora_text_rank,
                lora_alpha=self.params.lora_text_alpha,
                target_modules=self.TEXT_ENCODER_TARGET_MODULES,
                lora_dropout=self.params.lora_text_dropout,
            )
            text_encoder = LoraModel(lora_text_config, self._text_encoder())
        else:
            text_encoder = self._text_encoder()

        try:
            if self.accelerator.state.deepspeed_plugin:
                raise RuntimeError("DeepSpeed is not compatible with bitsandbytes.")

            import bitsandbytes as bnb
        except Exception as e:
            self._print(e)
            self._print("Could not import bitsandbytes, using AdamW")
            optimizer_class = torch.optim.AdamW
        else:
            optimizer_class = bnb.optim.AdamW8bit

        self._print("Initializing Optimizer...")

        lora_params = (
            itertools.chain(unet.parameters(), text_encoder.parameters())
            if self.params.train_text_encoder
            else unet.parameters()
        )

        optimizer = optimizer_class(
            lora_params,
            lr=self.params.learning_rate,
            betas=self.params.betas,
            weight_decay=self.params.weight_decay,
            eps=self.params.epsilon,
        )

        self._print("Loading dataset...")

        dataset = DreamBoothDataset(
            instance=self.instance_class,
            prior=self.params.prior_class or self.generate_priors(),
            tokenizer=self._tokenizer(),
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

        steps_per_epoch = math.ceil(
            len(loader) / self.params.gradient_accumulation_steps
        )
        max_train_steps = self.params.train_epochs * steps_per_epoch
        lr_scheduler = get_scheduler(
            self.params.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.params.lr_warmup_steps
            * self.params.gradient_accumulation_steps,
            num_training_steps=max_train_steps
            * self.params.gradient_accumulation_steps,
        )

        self._print("Preparing for training...")

        if self.params.train_text_encoder:
            (
                unet,
                text_encoder,
                optimizer,
                loader,
                lr_scheduler,
            ) = self.accelerator.prepare(
                unet, text_encoder, optimizer, loader, lr_scheduler
            )
        else:
            unet, optimizer, loader, lr_scheduler = self.accelerator.prepare(
                unet, optimizer, loader, lr_scheduler
            )

        steps_per_epoch = math.ceil(
            len(loader) / self.params.gradient_accumulation_steps
        )  # may have changed post-accelerate
        epochs = math.ceil(max_train_steps / steps_per_epoch)

        self._print(f"Training for {epochs} epochs. Max steps: {max_train_steps}.")

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers("dreambooth", config=self.params.dict())

        self._log_images(
            self.instance_class.prompt,
            map(str, self.instance_class.data.iterdir()),
            title="data",
        )

        self._print("Starting training...")

        models = {
            "unet": unet,
            "text_encoder": text_encoder,
            "vae": self._vae(),
            "noise_scheduler": self._noise_scheduler(),
            "lr_scheduler": lr_scheduler,
        }
        for epoch in range(epochs):
            self.logger.warning(f"Epoch {epoch + 1}/{epochs}", main_process_only=True)
            self._do_epoch(unet, loader, optimizer, models)
            if epoch % self.params.validate_every == 0:
                self._do_validation(unet, models)

        self.accelerator.wait_for_everyone()
        self._persist(unet, text_encoder)
        images = self._do_final_validation()
        self.accelerator.end_training()
        return images


def get_params() -> HyperParams:
    params = HyperParams()

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

    mem = torch.cuda.mem_get_info()[1] / 1e9
    print(f"Available GPU memory: {mem:.2f} GB")
    match mem:
        case float(n) if n < 16:
            params.batch_size = 1
            params.gradient_accumulation_steps = 2
            params.train_text_encoder = False
        case float(n) if n < 24:
            params.batch_size = 2
            params.gradient_accumulation_steps = 1
            params.train_text_encoder = True
        case float(n) if n < 32:
            params.batch_size = 4
            params.gradient_accumulation_steps = 1
            params.train_text_encoder = True
        case float(n):
            params.batch_size = 8 * torch.cuda.device_count()
            params.gradient_accumulation_steps = 1
            params.train_text_encoder = True

    match torch.cuda.device_count():
        case int(n) if n > 1:
            params.train_epochs //= n

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

    return Trainer(
        instance_class=Class(prompt="a photo of sks person", data=instance_path),
        params=params or get_params(),
    )
