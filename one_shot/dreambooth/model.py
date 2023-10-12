import PIL.Image
import torch
from controlnet_aux import (
    LineartDetector,
)
from diffusers import (
    AutoencoderKL,
    EulerAncestralDiscreteScheduler,
    StableDiffusionXLAdapterPipeline,
    T2IAdapter,
)


class LineartPreprocessor:
    def __init__(self):
        self.model = LineartDetector.from_pretrained("lllyasviel/Annotators")

    def to(self, device: torch.device | str):
        self.model.to(device)
        return self

    def __call__(self, image: PIL.Image.Image) -> PIL.Image.Image:
        return self.model(image, detect_resolution=384, image_resolution=1024)


class Model:
    MAX_NUM_INFERENCE_STEPS = 50

    def __init__(self, adapter_name: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = LineartPreprocessor().to(self.device)

        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        adapter = T2IAdapter.from_pretrained(
            "TencentARC/t2i-adapter-lineart-sdxl-1.0",
            torch_dtype=torch.float16,
            varient="fp16",
        ).to(self.device)
        self.pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
            model_id,
            vae=AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
            ),
            adapter=adapter,
            scheduler=EulerAncestralDiscreteScheduler.from_pretrained(
                model_id, subfolder="scheduler"
            ),
            torch_dtype=torch.float16,
            variant="fp16",
        ).to(self.device)
        self.pipe.enable_xformers_memory_efficient_attention()
        # self.pipe.load_lora_weights(
        #     "stabilityai/stable-diffusion-xl-base-1.0",
        #     weight_name="sd_xl_offset_example-lora_1.0.safetensors",
        # )
        # self.pipe.fuse_lora(lora_scale=0.4)

    def run(
        self,
        image: list[PIL.Image.Image],
        prompt: str,
        negative_prompt: str,
        adapter_name: str,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        adapter_conditioning_scale: float = 1.0,
        adapter_conditioning_factor: float = 1.0,
        seed: int = 0,
        apply_preprocess: bool = True,
    ) -> list[PIL.Image.Image]:
        image = [self.preprocessor(i) for i in image]

        # image = resize_to_closest_aspect_ratio(image)
        image = [i.resize((1024, 1024), PIL.Image.LANCZOS) for i in image]

        out = self.pipe(
            prompt=[prompt] * 2,
            negative_prompt=[negative_prompt] * 2,
            image=image,
            num_inference_steps=num_inference_steps,
            adapter_conditioning_scale=adapter_conditioning_scale,
            adapter_conditioning_factor=adapter_conditioning_factor,
            generator=torch.Generator(device=self.device).manual_seed(seed),
            guidance_scale=guidance_scale,
        ).images
        return out
