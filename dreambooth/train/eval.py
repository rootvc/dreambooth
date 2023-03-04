from pathlib import Path
from typing import TypeVar

import torch
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer

from dreambooth.params import HyperParams
from dreambooth.train.accelerators.base import BaseAccelerator

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
        model: T,
    ) -> T:
        print(f"Compiling {model.__class__.__name__}")
        return torch.compile(model, mode="max-autotune")

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

        return tokenizer, self._compile(text_encoder)

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
        return self._compile(unet)

    def _vae(self):
        vae = (
            AutoencoderKL.from_pretrained(self.params.model.vae)
            .to(self.accelerator.device)
            .requires_grad_(False)
            .eval()
        )
        return self._compile(vae)

    def _load_pipeline(self):
        tokenizer, text_encoder = self._init_text()

        pipeline = DiffusionPipeline.from_pretrained(
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

        pipeline = self._pipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
        )
        config = json.loads((self.output_dir / "lora_config.json").read_text())
        state = torch.load(
            self.output_dir / "lora_weights.pt", map_location=self.accelerator.device
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
        return self._validation(pipeline)

    @torch.inference_mode()
    def gen(self):
        pass
