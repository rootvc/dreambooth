from pathlib import Path
from typing import TypeVar

import torch
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
        print(f"Compiling {model.__class__.__name__} with {self.params.dynamo_backend}")
        return torch.compile(model, mode="max-autotune")

    def _load_pipeline(self):
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
            )
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
