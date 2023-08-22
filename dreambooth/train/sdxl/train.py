import itertools
import math
from typing import Optional, Tuple

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
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionXLControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
    get_scheduler,
)
from diffusers.loaders import (
    LoraLoaderMixin,
)
from diffusers.models.attention_processor import (
    LoRAXFormersAttnProcessor,
)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer

from dreambooth.params import Class
from dreambooth.train.base import BaseTrainer
from dreambooth.train.base import get_model as base_get_model
from dreambooth.train.datasets import (
    CachedLatentsDataset,
    DreamBoothDataset,
    PromptDataset,
)
from dreambooth.train.sdxl.utils import (
    encode_prompt,
    get_variance_type,
    import_model_class_from_model_name_or_path,
)
from dreambooth.train.shared import (
    dprint,
    patch_allowed_pipeline_classes,
    unpack_collate,
)
from dreambooth.train.utils import hash_image


class Trainer(BaseTrainer):
    def _text_encoders(self, **kwargs) -> Tuple[CLIPTextModel, CLIPTextModel]:
        klass_1 = import_model_class_from_model_name_or_path(
            self.params.model.name, self.params.model.revision
        )
        klass_2 = import_model_class_from_model_name_or_path(
            self.params.model.name,
            self.params.model.revision,
            subfolder="text_encoder_2",
        )

        te_1 = self._spawn(
            klass_1,
            subfolder="text_encoder",
            tap=lambda x: x.requires_grad_(False),
            **kwargs,
        )
        te_2 = self._spawn(
            klass_2,
            subfolder="text_encoder_2",
            tap=lambda x: x.requires_grad_(False),
            **kwargs,
        )

        te_1.to(self.accelerator.device, dtype=self.params.dtype, non_blocking=True)
        te_2.to(self.accelerator.device, dtype=self.params.dtype, non_blocking=True)
        return te_1, te_2

    def _text_encoders_for_lora(self, **kwargs):
        te_1, te_2 = self._text_encoders(**kwargs)
        params_1 = LoraLoaderMixin._modify_text_encoder(
            te_1, dtype=torch.float32, rank=self.params.lora_text_rank
        )
        params_2 = LoraLoaderMixin._modify_text_encoder(
            te_2, dtype=torch.float32, rank=self.params.lora_text_rank
        )
        return (te_1, te_2), (params_1, params_2)

    def _tokenizers(self) -> Tuple[CLIPTokenizer, CLIPTokenizer]:
        tok_1 = AutoTokenizer.from_pretrained(
            self.params.model.name,
            revision=self.params.model.revision,
            local_files_only=True,
            subfolder="tokenizer",
            use_fast=False,
        )
        tok_2 = AutoTokenizer.from_pretrained(
            self.params.model.name,
            revision=self.params.model.revision,
            local_files_only=True,
            subfolder="tokenizer_2",
            use_fast=False,
        )
        return tok_1, tok_2

    def _unet_for_lora(self, **kwargs):
        unet = super()._unet(**kwargs)
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

            module = LoRAXFormersAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=self.params.lora_rank,
            )
            unet_lora_attn_procs[name] = module
            unet_lora_parameters.extend(module.parameters())

        unet.set_attn_processor(unet_lora_attn_procs)
        return unet, unet_lora_parameters

    @torch.no_grad()
    def _pipeline(
        self,
        unet: Optional[UNet2DConditionModel] = None,
        text_encoders: Optional[Tuple[CLIPTextModel, CLIPTextModel]] = None,
        tokenizers: Optional[Tuple[CLIPTokenizer, CLIPTokenizer]] = None,
        vae: Optional[AutoencoderKL] = None,
        **kwargs,
    ) -> StableDiffusionXLControlNetPipeline:
        te_1, te_2 = text_encoders or [t.eval() for t in self._text_encoders()]
        tok_1, tok_2 = tokenizers or self._tokenizers()
        with patch_allowed_pipeline_classes():
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                self.params.model.name,
                revision=self.params.model.revision,
                local_files_only=True,
                safety_checker=None,
                vae=vae or self._vae().eval(),
                unet=(unet or self._unet()).eval(),
                text_encoder=te_1,
                text_encoder_2=te_2,
                tokenizer=tok_1,
                tokenizer_2=tok_2,
                controlnet=ControlNetModel.from_pretrained(
                    self.params.model.control_net,
                    local_files_only=True,
                    torch_dtype=self.params.dtype,
                    reset=True,
                ).to(self.accelerator.device),
                **kwargs,
            ).to(self.accelerator.device)
            pipe.enable_xformers_memory_efficient_attention()
            return pipe

    @torch.no_grad()
    def generate_priors(self, progress_bar: bool = False) -> Class:
        dprint("Generating priors...")

        pipeline = self._pipeline(torch_dtype=self.params.dtype)
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

    def _compute_time_ids(self):
        original_size = (self.params.model.resolution, self.params.model.resolution)
        target_size = (self.params.model.resolution, self.params.model.resolution)
        crops_coords_top_left = (0, 0)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        return add_time_ids.to(self.accelerator.device, dtype=self.params.dtype)

    def _load_models(self):
        with self.accelerator.init():
            unet, unet_params = self._unet_for_lora()
            text_encoders, te_params = self._text_encoders_for_lora()
            vae = self._vae()
            tokenizers = self._tokenizers()
        noise_scheduler = self._noise_scheduler()
        optimizer = self.accelerator.optimizer(
            itertools.chain(unet_params, *te_params),
            lr=self.params.learning_rate,
            betas=self.params.betas,
            weight_decay=self.params.weight_decay,
        )
        return (unet, text_encoders, vae, tokenizers, noise_scheduler, optimizer)

    def _prepare_models(self):
        (
            unet,
            text_encoders,
            vae,
            tokenizers,
            noise_scheduler,
            optimizer,
        ) = self._load_models()
        dataset = DreamBoothDataset(
            instance=self.instance_class,
            prior=self.params.prior_class or self.generate_priors(),
            tokenizers=tokenizers,
            size=self.params.model.resolution,
            vae_scale_factor=vae.config.scaling_factor,
        )

        dprint("Caching latents...")
        dataset = CachedLatentsDataset(self.accelerator, dataset, self.params, vae)
        warmed = dataset.warm()

        loader = DataLoader(
            warmed,
            batch_size=1,
            collate_fn=unpack_collate,
            shuffle=True,
            num_workers=self.params.loading_workers,
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
            tuple(text_encoders),
            vae,
            tokenizers,
            noise_scheduler,
            optimizer,
            loader,
            lr_scheduler,
            epochs,
        )

    def _do_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        noise_scheduler: DDPMScheduler,
        unet: UNet2DConditionModel,
        text_encoders: Tuple[CLIPTextModel, CLIPTextModel],
        scaling_factor: float,
        lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
        time_ids: torch.Tensor,
    ):
        unet.train()
        for text_encoder in text_encoders:
            text_encoder.train()

        for batch in loader:
            latent_dist, tokens = batch["latent_dist"], batch["tokens"]

            latents = latent_dist.sample()
            latents = latents * scaling_factor
            latents = latents.squeeze(0).to(self.accelerator.device).float()

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
                noise + self.params.input_perterbation * torch.randn_like(latents),
                timesteps,
            )

            # Calculate the elements to repeat depending on the use of prior-preservation.
            n_els = bsz // 2

            # Predict the noise residual
            prompt_embeds, pooled_prompt_embeds = encode_prompt(
                text_encoders=text_encoders,
                tokenizers=None,
                prompt=None,
                text_input_ids_list=tokens,
            )
            unet_added_conditions = {
                "time_ids": time_ids.repeat(n_els, 1),
                "text_embeds": pooled_prompt_embeds.repeat(n_els, 1),
            }
            model_pred = unet(
                noisy_latents,
                timesteps,
                prompt_embeds.repeat(n_els, 1, 1),
                added_cond_kwargs=unet_added_conditions,
            ).sample

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
                params_to_clip = optimizer.param_groups[0]["params"]
                self.accelerator.clip_grad_norm_(
                    params_to_clip, self.params.max_grad_norm
                )
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if self.accelerator.sync_gradients:
                self._total_steps += 1
            metrics = {
                "instance_loss": instance_loss.detach().item(),
                "prior_loss": prior_loss.detach().item(),
                "steps": self.total_steps,
                **self.metrics_cache,
            }
            self.accelerator.log(metrics)
            self.metrics_cache = {}

            if self.exceeded_max_steps():
                break

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
        ) = self._prepare_models()

        time_ids = torch.cat(
            [self._compute_time_ids(), self._compute_time_ids()], dim=0
        )

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
                time_ids,
            )
            if self.exceeded_max_steps():
                dprint("Max steps exceeded. Stopping training.")
                break

        self.accelerator.wait_for_everyone()

        pipe = self._pipeline(
            self.accelerator.unwrap_model(unet).eval(),
            tuple([self.accelerator.unwrap_model(te).eval() for te in text_encoders]),
            tokenizers,
            vae,
            torch_dtype=self.params.dtype,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config,
            disable_corrector=[0],
            **get_variance_type(pipe.scheduler),
        )
        return pipe.to(self.accelerator.device)


def get_model(**kwargs):
    return base_get_model(klass=Trainer, **kwargs)
