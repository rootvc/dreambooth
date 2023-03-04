from pathlib import Path

import torch
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline


class Evaluator:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.pipeline = self._load_pipeline()

    def _load_pipeline(self):
        token_embedding = torch.load(
            self.output_dir / "token_embedding.pt", map_location=self.accelerator.device
        )
        tokenizer, text_encoder = self._init_text(
            token_embedding[self.params.token], compile=True
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
