import json
from pathlib import Path
from typing import TypeVar

import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils import img2tensor, imwrite, tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.misc import get_device, gpu_is_available
from basicsr.utils.realesrgan_utils import RealESRGANer
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray
from peft import (
    LoraConfig,
    LoraModel,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer

from dreambooth.params import HyperParams
from dreambooth.train.accelerators.base import BaseAccelerator
from dreambooth.train.shared import partition

T = TypeVar("T", bound=torch.nn.Module)


class Evaluator:
    def __init__(
        self, accelerator: BaseAccelerator, params: HyperParams, model_dir: Path
    ):
        self.params = params
        self.accelerator = accelerator
        self.model_dir = model_dir

    def _compile(
        self,
        pipeline: StableDiffusionPipeline,
    ) -> StableDiffusionPipeline:
        pipeline.unet = torch.compile(pipeline.unet, mode="max-autotune")
        pipeline.vae = torch.compile(pipeline.vae, mode="max-autotune")
        pipeline.text_encoder = torch.compile(
            pipeline.text_encoder, mode="max-autotune"
        )
        return pipeline

    def _init_text(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.params.model.name,
            revision=self.params.model.revision,
            subfolder="tokenizer",
        )
        assert tokenizer.add_tokens(self.params.token) == 1

        text_encoder = (
            CLIPTextModel.from_pretrained(
                self.params.model.name,
                revision=self.params.model.revision,
                subfolder="text_encoder",
                dtype=self.params.dtype,
            )
            .to(self.accelerator.device)
            .requires_grad_(False)
            .eval()
        )
        text_encoder.resize_token_embeddings(len(tokenizer))

        token_id = tokenizer.convert_tokens_to_ids(self.params.token)
        embeds: torch.Tensor = text_encoder.get_input_embeddings().weight.data
        token_embedding = torch.load(
            self.model_dir / "token_embedding.pt", map_location=self.accelerator.device
        )

        embeds[token_id] = token_embedding[self.params.token]

        return tokenizer, text_encoder

    def _unet(self):
        unet = (
            UNet2DConditionModel.from_pretrained(
                self.params.model.name,
                revision=self.params.model.revision,
                subfolder="unet",
                dtype=self.params.dtype,
            )
            .to(self.accelerator.device)
            .requires_grad_(False)
            .eval()
        )
        return unet

    def _vae(self):
        vae = (
            AutoencoderKL.from_pretrained(self.params.model.vae)
            .to(self.accelerator.device)
            .requires_grad_(False)
            .eval()
        )
        return vae

    def _pipeline(self) -> StableDiffusionPipeline:
        tokenizer, text_encoder = self._init_text()
        return DiffusionPipeline.from_pretrained(
            self.params.model.name,
            revision=self.params.model.revision,
            safety_checker=None,
            low_cpu_mem_usage=True,
            local_files_only=True,
            unet=self._unet(),
            text_encoder=text_encoder,
            vae=self._vae(),
            tokenizer=tokenizer,
        )

    def _load_pipeline(self):
        pipeline = self._pipeline()

        config = json.loads((self.model_dir / "lora_config.json").read_text())
        state = torch.load(
            self.model_dir / "lora_weights.pt", map_location=self.accelerator.device
        )
        unet_state, text_state = partition(state, lambda kv: "text_encoder_" in kv[0])

        pipeline.unet = LoraModel(LoraConfig(**config["unet_peft"]), pipeline.unet).to(
            self.accelerator.device
        )
        set_peft_model_state_dict(pipeline.unet, unet_state)

        pipeline.text_encoder = LoraModel(
            LoraConfig(**config["text_peft"]), pipeline.text_encoder
        ).to(self.accelerator.device)
        set_peft_model_state_dict(
            pipeline.text_encoder,
            {k.removeprefix("text_encoder_"): v for k, v in text_state.items()},
        )

        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config
        )

        print("Compiling models...")
        return self._compile(pipeline)

    def _upsampler(self):
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )
        return RealESRGANer(
            scale=2,
            model_path=self.params.real_esrgan_path,
            model=model,
            pre_pad=0,
            half=True,
        )

    @torch.inference_mode()
    def gen(self):
        pipeline = self._load_pipeline()
        prompts = []
        images = pipeline(
            prompts,
            negative_prompt=self.params.negative_prompt,
            num_inference_steps=self.params.validation_steps,
        ).images
