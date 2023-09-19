from typing import Optional

import torch
from accelerate.utils import get_max_memory
from cloudpathlib import CloudPath
from pydantic import BaseModel, BaseSettings


class Settings(BaseSettings):
    bucket_name: str
    cache_dir: str

    @property
    def bucket(self) -> CloudPath:
        return CloudPath(self.bucket_name)

    def max_memory(self, device: Optional[int] = None):
        memory = get_max_memory()
        if device and device >= 0:
            return {
                device: memory[device],
                **{k: v for k, v in memory.items() if k != device},
            }
        else:
            return memory

    @property
    def loading_kwargs(self):
        return {
            "local_files_only": True,
            "use_safetensors": True,
            "low_cpu_mem_usage": True,
            "torch_dtype": torch.bfloat16,
        }


class Model(BaseModel):
    name: str = "stabilityai/stable-diffusion-xl-base-1.0"
    detector: str = "lllyasviel/Annotators"
    refiner: str = "stabilityai/stable-diffusion-xl-refiner-1.0"
    t2i_adapter: str = "TencentARC/t2i-adapter-lineart-sdxl-1.0"
    inpainter: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    loras: dict[str, dict[str, str]] = {
        "base": {name: "sd_xl_offset_example-lora_1.0.safetensors"},
        "inpainter": {"128461": "civitai"},
    }
    resolution: int = 1024


class Params(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    dtype: torch.dtype = torch.bfloat16
    model: Model = Model()
    batch_size: int = 4

    negative_prompt: str = ", ".join(
        [
            "out of focus",
            "lens blur",
            "low quality",
            "deformed eyes",
            "eyes closed",
            "distorted face",
            "disfigured",
            "airbrushed+",
        ]
    )
    negative_colors = [
        "purple----",
        "pink----",
        "green----",
        "brown----",
        "(low contrast)---",
    ]
    prompt_template = "closeup (4k photo)+ of a ({race})-- ({gender})-, ({prompt})++++, (cinematic camera)+, highly detailed, (ultra realistic)+, vibrant colors, high contrast, textured skin, realistic dull skin noise, visible skin detail, skin fuzz, dry skin"
    inpaint_prompt_template = "{color} eyes, perfecteyes++, (detailed pupils)+, subtle eyes, natural eyes, realistic eyes, ({race} {gender})-, ({prompt})--"
    prompts = [
        "(white face makeup)+, green hair, the joker, red nose, brilliant colors",
        "mysterious {race}, cyberpunk, the universe, cosmos and nebula on clothing",
        "(90s style)-, leather jacket, smug, vintage, antique car, distingushed, no makeup",
        "classy {gender}, wearing a rainbow suit, pop art style, painting by andy warhol",
        "zombie, (decaying skin and clothing)-, (rotting skin)-, inside an abandoned building",
        "handsome vampire, serious, black cape, pale skin, bright eyes, red lips, moonlit night",
        "(8-bit video game)+, pixelated+, minecraft, lego, blocky, colors of nature, farmer",
        "Marvel++ superhero+, superhero costume, flying in the air, sky+, nyc skyline in background, high contrast, simple colors",
        "a monarch, game of thrones, on the iron throne, wearing a crown, magestic, regal, powerful, bold",
        "rock star, face makeup, wearing a slick outfit, performing for fans, grungy, dark colors, moody",
        "character from tron, neon, techno, futuristic, dark background, black clothing, (high contrast)++",
    ]

    steps: int = 50
    inpainting_steps = 15
    images: int = 4

    detect_resolution: int = 384
    guidance_scale: float = 12.5
    refiner_strength = 0.30
    inpainting_strength = 0.80
    conditioning_strength: tuple[float, float] = (2.28, 2.35)
    conditioning_factor: float = 1.0
    lora_scale = 0.3
    high_noise_frac: float = 0.90
    mask_padding: float = 0.05
