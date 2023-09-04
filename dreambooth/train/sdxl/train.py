import itertools
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
import torch.utils.checkpoint
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import (
    LoraLoaderMixin,
)
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer

from dreambooth.params import Class
from dreambooth.train.base import BaseTrainer
from dreambooth.train.base import get_model as base_get_model
from dreambooth.train.sdxl.utils import (
    encode_prompt,
    import_model_class_from_model_name_or_path,
)
from dreambooth.train.shared import (
    dprint,
    patch_allowed_pipeline_classes,
)


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
            subfolder="tokenizer",
            use_fast=False,
        )
        tok_2 = AutoTokenizer.from_pretrained(
            self.params.model.name,
            revision=self.params.model.revision,
            subfolder="tokenizer_2",
            use_fast=False,
        )
        return tok_1, tok_2

    @torch.no_grad()
    def _pipeline(
        self,
        unet: Optional[UNet2DConditionModel] = None,
        text_encoders: Optional[Tuple[CLIPTextModel, CLIPTextModel]] = None,
        tokenizers: Optional[Tuple[CLIPTokenizer, CLIPTokenizer]] = None,
        vae: Optional[AutoencoderKL] = None,
        **kwargs,
    ) -> StableDiffusionXLControlNetPipeline:
        te_1, te_2 = text_encoders or [
            t.eval().to(dtype=self.params.dtype) for t in self._text_encoders()
        ]
        tok_1, tok_2 = tokenizers or self._tokenizers()
        with patch_allowed_pipeline_classes():
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                self.params.model.name,
                revision=self.params.model.revision,
                safety_checker=None,
                vae=(vae or self._vae()).eval().to(dtype=torch.float32),
                unet=(unet or self._unet()).eval().to(dtype=self.params.dtype),
                text_encoder=te_1,
                text_encoder_2=te_2,
                tokenizer=tok_1,
                tokenizer_2=tok_2,
                controlnet=ControlNetModel.from_pretrained(
                    self.params.model.control_net,
                    reset=True,
                ).to(self.accelerator.device, dtype=self.params.dtype),
                **kwargs,
            ).to(self.accelerator.device, torch_dtype=self.params.dtype)
            pipe.enable_xformers_memory_efficient_attention()
            return pipe

    def generate_priors(self) -> Class:
        dprint("Generating priors...")
        return super()._generate_priors_with(StableDiffusionXLPipeline)

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
        params = list(itertools.chain(unet_params, *te_params))
        optimizer = self.accelerator.optimizer(
            params,
            lr=self.params.learning_rate,
            betas=self.params.betas,
            weight_decay=self.params.weight_decay,
        )
        return (
            unet,
            text_encoders,
            vae,
            tokenizers,
            noise_scheduler,
            optimizer,
            params,
        )

    def _unet_epoch_args(self, batch: dict, bsz: int, text_encoders: list):
        n_els = bsz // 2
        all_prompt_embeds, all_pooled_prompt_embeds = zip(
            *[
                encode_prompt(
                    text_encoders=text_encoders,
                    tokenizers=None,
                    prompt=None,
                    text_input_ids_list=t,
                )
                for t in batch["tokens"]
            ]
        )
        time_ids = torch.cat(
            [self._compute_time_ids(), self._compute_time_ids()], dim=0
        )
        unet_added_conditions = {
            "time_ids": time_ids.repeat(n_els, 1),
            "text_embeds": torch.concat(all_pooled_prompt_embeds),
        }

        return [torch.concat(all_prompt_embeds)], {
            "added_cond_kwargs": unet_added_conditions
        }

    def _unwrap_pipe_args(
        self, unet, text_encoders, tokenizers, vae
    ) -> tuple[list, dict]:
        return [
            self.accelerator.unwrap_model(unet, False).eval(),
            tuple(
                [
                    self.accelerator.unwrap_model(te, False).eval()
                    for te in text_encoders
                ]
            ),
            tokenizers,
            vae,
        ], {}


def get_model(**kwargs):
    return base_get_model(klass=Trainer, **kwargs)
