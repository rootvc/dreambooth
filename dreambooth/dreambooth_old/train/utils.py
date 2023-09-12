import itertools
import json
import math
import sys
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from queue import Empty
from threading import Event
from typing import Any, Callable, Optional, Type, TypeVar, cast

import numpy as np
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
import torch.multiprocessing as multiprocessing
import torch.nn.functional as F
import wandb
from accelerate.logging import get_logger
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DPMSolverMultistepScheduler,
    StableDiffusionControlNetPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
    get_scheduler,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import AttnProcessor2_0
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from PIL import Image
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    CLIPTextModel,
    CLIPTokenizer,
)
from transformers import pipeline as transformers_pipeline

from dreambooth_old.params import Class
from dreambooth_old.registry import CompiledModelsRegistry
from dreambooth_old.train.accelerators import BaseAccelerator
from dreambooth_old.train.base import BaseTrainer
from dreambooth_old.train.datasets import (
    CachedLatentsDataset,
    DreamBoothDataset,
    PromptDataset,
)
from dreambooth_old.train.shared import (
    Mode,
    dprint,
    hash_image,
    images,
    main_process_only,
    partition,
    patch_allowed_pipeline_classes,
    unpack_collate,
)

T = TypeVar("T")


torch.backends.cudnn.benchmark = True
torch._dynamo.config.suppress_errors = True

get_logger("torch._dynamo.eval_frame").setLevel("ERROR")


class ScoreThresholdExceeded(Exception):
    pass


class TrainingProcess:
    METHOD = "forkserver"
    TIMEOUT = 60 * 3

    process: multiprocessing.ProcessContext

    @staticmethod
    def _loop(
        _i,
        recv: multiprocessing.Queue,
        send: multiprocessing.Queue,
        event: Event,
    ):
        while True:
            fn, args = recv.get()
            if args is None:
                break
            result = fn(*args)
            send.put_nowait(result)
            event.wait(10)
            event.clear()

    def __init__(self) -> None:
        ctx = multiprocessing.get_context(self.METHOD)
        self.send = ctx.Queue(1)
        self.recv = ctx.Queue(1)
        self.event = ctx.Event()
        self.process = None

    def start(self) -> None:
        if self.process:
            raise RuntimeError("Process already started")
        self.process = multiprocessing.start_processes(
            self._loop,
            args=(self.send, self.recv, self.event),
            join=False,
            start_method=self.METHOD,
        )

    def __call__(self, fn: Callable, *args):
        self.send.put_nowait((fn, args))

        def wait(extract: Callable):
            try:
                response = self.recv.get(timeout=self.TIMEOUT)
            except Empty:
                self.process.join(1)
                raise

            result = extract(response)
            self.event.set()
            return result

        return wait


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


class Trainer(BaseTrainer):
    UNET_TARGET_MODULES = ["to_q", "to_v", "query", "value"]
    TEXT_ENCODER_TARGET_MODULES = ["q_proj", "v_proj"]

    accelerator: BaseAccelerator

    def _spawn(
        self, klass: Type[T], tap: Callable[[T], Any] = lambda x: x, **kwargs
    ) -> T:
        instance = CompiledModelsRegistry.get(
            klass,
            self.params.model.name,
            revision=self.params.model.revision,
            local_files_only=True,
            **kwargs,
        )
        tap(instance)
        return instance

    @classmethod
    def _process(cls):
        if not hasattr(cls, "__process"):
            cls.__process = TrainingProcess()
            cls.__process.start()
        return cls.__process

    @lru_cache(maxsize=1)
    def token_id(self, tokenizer: CLIPTokenizer):
        return tokenizer.convert_tokens_to_ids(self.params.token)

    @torch.no_grad()
    def _pipeline(
        self,
        unet: Optional[UNet2DConditionModel] = None,
        text_encoder: Optional[CLIPTextModel] = None,
        vae: Optional[AutoencoderKL] = None,
        tokenizer: Optional[CLIPTokenizer] = None,
        **kwargs,
    ) -> StableDiffusionControlNetPipeline:
        with patch_allowed_pipeline_classes():
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                self.params.model.name,
                revision=self.params.model.revision,
                local_files_only=True,
                safety_checker=None,
                vae=vae or self._vae(compile=True).eval(),
                unet=(unet or self._unet(mode=Mode.TI, compile=True)).eval(),
                text_encoder=(
                    (text_encoder or self._text_encoder(compile=True)).eval()
                ),
                tokenizer=tokenizer or self._tokenizer(),
                controlnet=CompiledModelsRegistry.get(
                    ControlNetModel,
                    self.params.model.control_net,
                    compile=True,
                    local_files_only=True,
                    torch_dtype=torch.float,
                    reset=True,
                ).to(self.accelerator.device),
                **kwargs,
            )
        if "device_map" not in kwargs:
            pipe.to(self.accelerator.device)
        # pipe.enable_xformers_memory_efficient_attention()
        return pipe

    def _text_encoder(self, compile: bool = False, **kwargs) -> CLIPTextModel:
        te = self._spawn(
            CLIPTextModel,
            subfolder="text_encoder",
            tap=lambda x: x.requires_grad_(False),
            compile=compile,
            **kwargs,
        )
        if "device_map" not in kwargs:
            te.to(self.accelerator.device, non_blocking=True)
        return te

    def _tokenizer(self) -> CLIPTokenizer:
        return AutoTokenizer.from_pretrained(
            self.params.model.name,
            revision=self.params.model.revision,
            local_files_only=True,
            subfolder="tokenizer",
            use_fast=False,
        )

    def _unet_config(self) -> LoraConfig:
        return LoraConfig(
            r=self.params.lora_rank,
            # init_r=int(self.params.lora_rank * 1.5),
            # tinit=self.params.lr_warmup_steps,
            lora_alpha=self.params.lora_alpha,
            target_modules=self.UNET_TARGET_MODULES,
            lora_dropout=self.params.lora_dropout,
        )

    def _unet(
        self,
        compile: bool = False,
        mode: Mode = Mode.LORA,
        **kwargs,
    ) -> UNet2DConditionModel:
        unet: UNet2DConditionModel = self._spawn(
            UNet2DConditionModel,
            subfolder="unet",
            tap=lambda x: x.requires_grad_(False),
            compile=compile and mode != Mode.LORA,
            mode=mode,
            **kwargs,
        )
        if "device_map" not in kwargs:
            unet.to(self.accelerator.device, non_blocking=True)

        if mode == Mode.LORA:
            unet.set_attn_processor(AttnProcessor2_0())
            lora_config = self._unet_config()
            unet = CompiledModelsRegistry.wrap(
                get_peft_model(
                    CompiledModelsRegistry.unwrap(unet),
                    lora_config,
                )
            )
            if "device_map" not in kwargs:
                unet.to(self.accelerator.device, non_blocking=True)

            unet = compile_model(unet, do=compile)

        return unet

    @torch.no_grad()
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
        # torch.cuda.empty_cache()

        return Class(prompt_=self.params.prior_prompt, data=self.priors_dir)

    @torch.no_grad()
    def generate_depth_values(self, path: Path) -> float:
        dprint("Generating depth values...")

        depth_estimator = transformers_pipeline(
            "depth-estimation",
            torch_dtype=self.params.dtype,
        )

        for img_path in (progress := tqdm(images(path))):
            depth_path = DreamBoothDataset.depth_image_path(img_path)
            # if depth_path.exists():
            #     continue
            progress.set_description(f"Processing {img_path.name}")
            image = DreamBoothDataset.open_image(img_path)
            depth = depth_estimator(image)["depth"]
            depth = np.array(depth)
            depth = depth[:, :, None]
            depth = np.concatenate([depth, depth, depth], axis=2)
            depth_image = Image.fromarray(depth)
            depth_image.save(depth_path)

    @main_process_only
    @torch.no_grad()
    def _validation(
        self, pipeline: StableDiffusionPipeline, final: bool = False, **kwargs
    ):
        score = self.tester.test_pipe(
            pipeline, ("final_validation" if final else "validation") + str(kwargs)
        )
        # torch.cuda.empty_cache()

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
        input_embeddings: torch.Tensor,
        check: bool = False,
    ):
        dprint("Final validation...")
        pipeline = self._create_eval_model(
            unet,
            text_encoder,
            input_embeddings,
            self.params.lora_alpha,
            self.params.lora_text_alpha,
        )
        if check:
            try:
                self._validation(
                    pipeline,
                    final=True,
                    alpha=self.params.lora_alpha,
                    text_alpha=self.params.lora_text_alpha,
                )
            except ScoreThresholdExceeded:
                pass
            else:
                raise RuntimeError("Final validation failed")
            finally:
                self._total_steps += 1
                wandb.run.log(self.metrics_cache, step=self.total_steps, commit=True)
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
                # torch.cuda.empty_cache()

    @torch.no_grad()
    def _create_eval_model(
        self,
        unet_0: UNet2DConditionModel,
        text_encoder_0: CLIPTextModel,
        input_embeddings: torch.Tensor,
        alpha: float,
        alpha_txt: float,
    ):
        unet_0.peft_config["default"] = self._unet_config()
        unet_state = get_peft_model_state_dict(
            unet_0, state_dict=self.accelerator.get_state_dict(unet_0)
        )
        unet_config = unet_0.get_peft_config_as_dict(inference=True)
        unet_config["lora_alpha"] = alpha
        unet_config["lora_dropout"] = sys.float_info.epsilon

        unet = CompiledModelsRegistry.unwrap(
            self._unet(mode=Mode.TI, compile=False, reset=True)
            .requires_grad_(False)
            .eval()
        )
        unet = cast(
            UNet2DConditionModel,
            (
                get_peft_model(
                    self.accelerator.unwrap_model(unet), LoraConfig(**unet_config)
                )
                .to(unet_0.device, non_blocking=True)
                .requires_grad_(False)
            ),
        )
        set_peft_model_state_dict(unet, unet_state)
        unet = compile_model(CompiledModelsRegistry.wrap(unet.merge_and_unload()))

        tokenizer = self._tokenizer()
        tokenizer.add_tokens(self.params.token)
        text_encoder = CompiledModelsRegistry.unwrap(
            self._text_encoder(compile=False, reset=True).eval().requires_grad_(False)
        )
        text_encoder.resize_token_embeddings(len(tokenizer))

        embeds: torch.Tensor = text_encoder.get_input_embeddings().weight.data
        token_id = self.token_id(tokenizer)
        embeds[token_id] = input_embeddings[token_id]

        text_encoder_0.peft_config["default"] = self._text_config()
        text_state = get_peft_model_state_dict(
            text_encoder_0, state_dict=self.accelerator.get_state_dict(text_encoder_0)
        )
        text_config = text_encoder_0.get_peft_config_as_dict(inference=True)
        text_config["lora_alpha"] = alpha_txt
        text_config["lora_dropout"] = sys.float_info.epsilon

        text_encoder = cast(
            CLIPTextModel,
            (
                get_peft_model(
                    self.accelerator.unwrap_model(text_encoder),
                    LoraConfig(**text_config),
                )
                .to(text_encoder_0.device, non_blocking=True)
                .requires_grad_(False)
            ),
        )
        set_peft_model_state_dict(text_encoder, text_state)
        text_encoder = compile_model(
            CompiledModelsRegistry.wrap(text_encoder.merge_and_unload())
        )

        vae = compile_model(
            CompiledModelsRegistry.unwrap(self._vae(compile=False)).eval()
        )

        pipeline = self._pipeline(
            unet=unet.to(dtype=torch.float),
            text_encoder=text_encoder.to(dtype=torch.float),
            tokenizer=tokenizer,
            vae=vae.to(dtype=torch.float),
            torch_dtype=torch.float,
        )
        pipeline.scheduler = UniPCMultistepScheduler.from_config(
            pipeline.scheduler.config, disable_corrector=[0]
        )
        return pipeline.to(self.accelerator.device)

    @torch.no_grad()
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
                    get_peft_model(unet, LoraConfig(**config["unet_peft"]))
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

        text_encoder = get_peft_model(
            text_encoder, LoraConfig(**config["text_peft"])
        ).to(self.accelerator.device, non_blocking=True)
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

    # @torch.compile()
    @torch.no_grad()
    def _compute_snr(self, models, timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = models["noise_scheduler"].alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
            timesteps
        ].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
            device=timesteps.device
        )[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        return (alpha / sigma) ** 2

    def _do_epoch(
        self,
        mode: Mode,
        unet: UNet2DConditionModel,
        loader: DataLoader,
        optimizer: Optimizer,
        models: dict,
    ):
        unet.train()
        models["text_encoder"].train()

        for batch in loader:
            # Convert images to latent space
            latents = batch["latents"]

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
                latents,
                noise + self.params.input_perterbation * torch.randn_like(latents),
                timesteps,
            )
            # noisy_latents = torch.cat([noisy_latents, depth_values], dim=1)

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

            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            # snr = self._compute_snr(models, timesteps)
            # loss_weights = (
            #     torch.stack(
            #         [snr, self.params.snr_gamma * torch.ones_like(timesteps)], dim=1
            #     ).min(dim=1)[0]
            #     / snr
            # )

            # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
            model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
            target, target_prior = torch.chunk(target, 2, dim=0)
            # loss_weights, loss_weights_prior = torch.chunk(loss_weights, 2, dim=0)

            # We first calculate the original loss. Then we mean over the non-batch dimensions and
            # rebalance the sample-wise losses with their respective loss weights.
            # Finally, we take the mean of the rebalanced loss.

            # Compute instance loss
            instance_loss = (
                F.mse_loss(model_pred.float(), target.float(), reduction="none").mean(
                    [1, 2, 3]
                )
                # .mul(loss_weights)
                .mean()
            )

            # Compute prior loss
            prior_loss = (
                F.mse_loss(
                    model_pred_prior.float(), target_prior.float(), reduction="none"
                ).mean([1, 2, 3])
                # .mul(loss_weights_prior)
                .mean()
            )

            # Add the prior loss to the instance loss.
            loss = instance_loss + self.params.prior_loss_weight * prior_loss

            self.accelerator.backward(loss)

            if mode == Mode.TI:
                self._do_textual_inversion(models)

            if self.accelerator.sync_gradients:
                params_to_clip = itertools.chain(
                    models["unet_params"],
                    models["text_encoder"].parameters(),
                )
                self.accelerator.clip_grad_norm_(
                    params_to_clip, self.params.max_grad_norm
                )
            optimizer.step()
            optimizer.zero_grad()

            models["lr_scheduler"].step()

            if self.accelerator.sync_gradients:
                self._total_steps += 1

            metrics = {
                f"{mode.name}_instance_loss": instance_loss.detach().item(),
                f"{mode.name}_prior_loss": prior_loss.detach().item(),
                f"{mode.name}_steps": self.total_steps,
                **self.metrics_cache,
            }

            if mode == Mode.TI:
                metrics["ti_lr"] = models["lr_scheduler"].get_last_lr()[0]
            elif mode == Mode.LORA:
                metrics["ti_lr"] = models["lr_scheduler"].get_last_lr()[0]
                metrics["text_lr"] = models["lr_scheduler"].get_last_lr()[1]
                metrics["unet_lr"] = models["lr_scheduler"].get_last_lr()[2]

            self.accelerator.log(metrics)
            self.metrics_cache = {}

            if self.exceeded_max_steps():
                break

    @torch.no_grad()
    def _do_textual_inversion(self, models: dict):
        grad = models["text_encoder"].get_input_embeddings().weight.grad
        ids = torch.arange(len(models["tokenizer"]))
        tk_id = self.token_id(models["tokenizer"])
        idx = ids != tk_id
        grad.data[idx, :] = grad.data[idx, :].fill_(0)

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

    def _text_config(self):
        return LoraConfig(
            r=self.params.lora_text_rank,
            # init_r=int(self.params.lora_text_rank * 1.5),
            lora_alpha=self.params.lora_text_alpha,
            # tinit=self.params.lr_warmup_steps,
            target_modules=self.TEXT_ENCODER_TARGET_MODULES,
            lora_dropout=self.params.lora_text_dropout,
        )

    @torch.no_grad()
    def _init_text(
        self,
        init: Optional[torch.Tensor] = None,
        compile: bool = False,
        mode: Mode = Mode.LORA,
        **kwargs,
    ):
        tokenizer = self._tokenizer()
        tokenizer.add_tokens(self.params.token)

        text_encoder: CLIPTextModel = self._text_encoder(
            compile=compile and mode != Mode.LORA, **kwargs
        )
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

        if mode == Mode.LORA:
            lora_text_config = self._text_config()
            text_encoder = CompiledModelsRegistry.wrap(
                get_peft_model(
                    CompiledModelsRegistry.unwrap(text_encoder),
                    lora_text_config,
                )
            )
            text_encoder = compile_model(text_encoder, do=compile)

        text_encoder = text_encoder.to(self.accelerator.device, non_blocking=True)
        return (tokenizer, text_encoder)

    def _unet_param_source(self, unet: UNet2DConditionModel):
        if self.params.use_diffusers_unet:
            lora_layers = AttnProcsLayers(unet.attn_processors)
            self.accelerator.register_for_checkpointing(lora_layers)
            return lora_layers
        else:
            return unet

    @torch.no_grad()
    def _prepare_dataset(self):
        tokenizer = self._tokenizer()
        tokenizer.add_tokens(self.params.token)

        vae = self._vae(compile=True, reset=True, torch_dtype=self.params.dtype)
        # self.generate_depth_values(self.instance_class.data)
        dataset = DreamBoothDataset(
            instance=self.instance_class,
            prior=self.params.prior_class or self.generate_priors(),
            tokenizer=tokenizer,
            size=self.params.model.resolution,
            vae_scale_factor=self._vae().config.scaling_factor,
        )
        # self.generate_depth_values(dataset.prior.data)
        dprint("Caching latents...")
        dataset = CachedLatentsDataset(self.accelerator, dataset, self.params, vae)
        warmed = dataset.warm()

        return {"dataset": warmed, "tokenizer": tokenizer}

    def _load_models(self, models: dict, mode: Mode):
        with self.accelerator.init():
            tokenizer, text_encoder = self._init_text(
                compile=mode != Mode.LORA,
                mode=mode,
                reset=True,
                torch_dtype=torch.float,
            )
            unet = self._unet(
                compile=mode != Mode.LORA,
                wrap=mode != Mode.LORA,
                mode=mode,
                reset=True,
                torch_dtype=torch.float,
            )

        return {
            **models,
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "unet": unet,
        }

    def _prepare_models(self, models: dict, mode: Mode):
        tokenizer, unet, text_encoder = (
            models["tokenizer"],
            models["unet"],
            models["text_encoder"],
        )

        loader = DataLoader(
            models["dataset"],
            batch_size=1,
            collate_fn=unpack_collate,
            shuffle=True,
            num_workers=0,
        )

        ti_params = [
            p.requires_grad_(
                mode == Mode.TI or self.params.ti_continued_learning_rate > 0.0
            )
            for p in text_encoder.get_input_embeddings().parameters()
        ]
        text_params = list(set(text_encoder.parameters()) - set(ti_params))
        unet_params = self._unet_param_source(unet).parameters()

        if mode == Mode.TI:
            params = [
                {"lr": self.params.ti_learning_rate, "params": ti_params},
            ]
            epochs = self.params.ti_train_epochs
        elif mode == Mode.LORA:
            params = [
                {"lr": self.params.ti_continued_learning_rate, "params": ti_params},
                {"lr": self.params.text_learning_rate, "params": text_params},
                {"lr": self.params.learning_rate, "params": unet_params},
            ]
            epochs = self.params.lora_train_epochs
        else:
            raise ValueError(f"Unknown mode {mode}")

        steps_per_epoch = math.ceil(
            len(loader)
            * self.params.batch_size
            / self.params.gradient_accumulation_steps
        )
        max_train_steps = epochs * steps_per_epoch

        if mode == Mode.LORA:
            unet.base_model.peft_config["default"].total_step = max_train_steps
            text_encoder.base_model.peft_config["default"].total_step = max_train_steps

        # loader = self.accelerator.prepare(loader)
        optimizer = self.accelerator.optimizer(
            params,
            betas=self.params.betas,
            weight_decay=self.params.weight_decay,
        )

        # optimizer = self.accelerator.prepare(optimizer)

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

        vae = self._vae(compile=True)

        return {
            "epochs": epochs,
            "max_train_steps": max_train_steps,
            "unet": unet,
            "text_encoder": text_encoder,
            "loader": loader,
            "tokenizer": tokenizer,
            "optimizer": optimizer,
            "vae": vae,
            "lr_scheduler": lr_scheduler,
            "noise_scheduler": self._noise_scheduler(),
            "unet_params": unet_params,
        }

    def _train(self, models: dict, mode: Mode):
        dprint(f"Starting {mode} training...")
        unet, text_encoder, epochs, optimizer, loader, lr_scheduler = (
            models["unet"],
            models["text_encoder"],
            models["epochs"],
            models["optimizer"],
            models["loader"],
            models["lr_scheduler"],
        )

        for epoch in range(epochs):
            if self.accelerator.is_main_process:
                dprint(
                    f"[{mode.name}] Epoch {epoch + 1}/{epochs} (Step {self.total_steps})"
                )

            pg, blr = optimizer.param_groups, lr_scheduler.base_lrs
            if mode == Mode.TI:
                pg[0]["lr"] = blr[0] = self.params.ti_learning_rate
            elif mode == Mode.LORA:
                pg[0]["lr"] = blr[0] = self.params.ti_continued_learning_rate
                pg[1]["lr"] = blr[1] = self.params.text_learning_rate
                pg[2]["lr"] = blr[2] = self.params.learning_rate

            self._do_epoch(mode, unet, loader, optimizer, models)
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

        # self.accelerator.wait_for_everyone()
        unet = CompiledModelsRegistry.unwrap(unet)
        text_encoder = CompiledModelsRegistry.unwrap(text_encoder)

        return unet, text_encoder

    def _train_sequentially(self):
        dataset = self._prepare_dataset()

        models = self._prepare_models(self._load_models(dataset, Mode.TI), Mode.TI)
        _, text_encoder = self._train(models, Mode.TI)
        input_embeddings = text_encoder.get_input_embeddings().weight.data

        models = self._load_models(dataset, Mode.LORA)
        token_id = self.token_id(models["tokenizer"])
        models["text_encoder"].get_input_embeddings().weight.data[
            token_id
        ] = input_embeddings[token_id]

        models = self._prepare_models(models, Mode.LORA)

        unet, text_encoder = self._train(models, Mode.LORA)
        input_embeddings = text_encoder.eval().get_input_embeddings().weight.data

        return (unet, text_encoder, input_embeddings)

    def _init_and_train(self, dataset: dict, mode: Mode):
        models = self._prepare_models(self._load_models(dataset, mode), mode)
        dprint(
            f"Training for {models['epochs']} epochs. Max steps: {models['max_train_steps']}."
        )
        unet, text_encoder = self._train(models, mode)
        return {
            "unet": CompiledModelsRegistry.unwrap(unet),
            "text_encoder": CompiledModelsRegistry.unwrap(text_encoder),
        }

    def _spawn_train_mode(self, *args):
        accelerator = self._accelerator()
        accelerator.trackers = self.accelerator.trackers
        self.accelerator = accelerator
        return self._init_and_train(*args)

    def _train_and_combine(self):
        dataset = self._prepare_dataset()

        get_ti_models = self._process()(self._spawn_train_mode, dataset, Mode.TI)
        lora_models = self._init_and_train(dataset, Mode.LORA)
        input_embeddings = get_ti_models(
            lambda models: models["text_encoder"]
            .eval()
            .get_input_embeddings()
            .weight.data.clone()
        )

        return (lora_models["unet"], lora_models["text_encoder"], input_embeddings)

    def train(self):
        # self.accelerator.free_memory()
        self._prepare_to_train()
        # unet, text_encoder, input_embeddings = self._train_and_combine()
        unet, text_encoder, input_embeddings = self._train_sequentially()
        # self.accelerator.free_memory()
        # self.accelerator.wait_for_everyone()
        return self._do_final_validation(
            unet, text_encoder, input_embeddings, check=False
        )