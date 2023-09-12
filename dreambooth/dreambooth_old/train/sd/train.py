import itertools
from typing import Optional

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
    StableDiffusionControlNetPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import (
    LoraLoaderMixin,
)
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer

from dreambooth_old.params import Class
from dreambooth_old.train.base import BaseTrainer
from dreambooth_old.train.base import get_model as base_get_model
from dreambooth_old.train.sdxl.utils import (
    import_model_class_from_model_name_or_path,
)
from dreambooth_old.train.shared import (
    dprint,
    patch_allowed_pipeline_classes,
)


class Trainer(BaseTrainer):
    def _text_encoder(self, **kwargs) -> CLIPTextModel:
        klass = import_model_class_from_model_name_or_path(
            self.params.model.name, self.params.model.revision
        )
        text_encoder = self._spawn(
            klass,
            subfolder="text_encoder",
            tap=lambda x: x.requires_grad_(False),
            **kwargs,
        )
        return text_encoder.to(
            self.accelerator.device, dtype=self.params.dtype, non_blocking=True
        )

    def _text_encoder_for_lora(self, **kwargs):
        te = self._text_encoder(**kwargs)
        params = LoraLoaderMixin._modify_text_encoder(
            te, dtype=torch.float32, rank=self.params.lora_text_rank
        )
        return te, params

    def _tokenizer(self) -> CLIPTokenizer:
        return AutoTokenizer.from_pretrained(
            self.params.model.name,
            revision=self.params.model.revision,
            subfolder="tokenizer",
            use_fast=False,
        )

    @torch.inference_mode()
    def _pipeline(
        self,
        unet: Optional[UNet2DConditionModel] = None,
        text_encoder: Optional[CLIPTextModel] = None,
        tokenizer: Optional[CLIPTokenizer] = None,
        vae: Optional[AutoencoderKL] = None,
        klass: type = StableDiffusionPipeline,
        **kwargs,
    ) -> StableDiffusionControlNetPipeline:
        with patch_allowed_pipeline_classes():
            pipe = klass.from_pretrained(
                self.params.model.name,
                revision=self.params.model.revision,
                safety_checker=None,
                vae=(vae or self._vae()).eval(),
                unet=(unet or self._unet()).eval(),
                text_encoder=(text_encoder or self._text_encoder()).eval(),
                tokenizer=tokenizer or self._tokenizer(),
                controlnet=ControlNetModel.from_pretrained(
                    self.params.model.control_net,
                    reset=True,
                ).to(self.accelerator.device, dtype=self.params.dtype),
                **kwargs,
            ).to(self.accelerator.device, torch_dtype=self.params.dtype)
            # pipe.enable_xformers_memory_efficient_attention()
            # pipe.fuse_lora(self.params.lora_alpha)
            return pipe

    def generate_priors(self) -> Class:
        dprint("Generating priors...")
        return super()._generate_priors_with(StableDiffusionPipeline)

    def _load_models(self):
        with self.accelerator.init():
            unet, unet_params = self._unet_for_lora()
            text_encoder, te_params = self._text_encoder_for_lora()
            vae = self._vae()
            tokenizer = self._tokenizer()
        noise_scheduler = self._noise_scheduler()
        params = [
            {"lr": self.params.learning_rate, "params": unet_params},
            {"lr": self.params.text_learning_rate, "params": te_params},
        ]
        optimizer = self.accelerator.optimizer(
            params,
            betas=self.params.betas,
            weight_decay=self.params.weight_decay,
        )
        return (
            unet,
            [text_encoder],
            vae,
            [tokenizer],
            noise_scheduler,
            optimizer,
            itertools.chain.from_iterable(p["params"] for p in params),
        )

    def _unet_epoch_args(self, batch: dict, bsz: int, text_encoders: list):
        tokens = torch.cat([t[0] for t in batch["tokens"]], dim=0).to(
            self.accelerator.device, dtype=torch.long
        )
        prompt_embeds = text_encoders[0](tokens)[0]
        return [prompt_embeds], {}

    def _unwrap_pipe_args(
        self, unet, text_encoders, tokenizers, vae
    ) -> tuple[list, dict]:
        return [
            self.accelerator.unwrap_model(unet, False).eval(),
            self.accelerator.unwrap_model(text_encoders[0], False).eval(),
            tokenizers[0],
            vae,
        ], {}


def get_model(**kwargs):
    return base_get_model(klass=Trainer, **kwargs)
