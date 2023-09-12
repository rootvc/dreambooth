from diffusers import (
    DiffusionPipeline,
    StableDiffusionXLPipeline,
)


class StableDiffusionXLEnsemblePipeline(StableDiffusionXLPipeline):
    def __call__(
        self,
        *args,
        refiner_name: str,
        prompt: str | list[str],
        num_inference_steps: int = 50,
        high_noise_frac: float,
        **kwargs,
    ):
        if not hasattr(self, "refiner"):
            self.refiner = DiffusionPipeline.from_pretrained(
                refiner_name,
                text_encoder_2=self.text_encoder_2,
                vae=self.vae,
            ).to(self.device, torch_dtype=self.text_encoder_2.dtype)

        latents = (
            super()
            .__call__(
                *args,
                prompt=prompt,
                denoising_end=high_noise_frac,
                num_inference_steps=num_inference_steps,
                output_type="latent",
                **kwargs,
            )
            .images
        )
        return self.refiner(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            denoising_start=high_noise_frac,
            image=latents,
            **kwargs,
        )
