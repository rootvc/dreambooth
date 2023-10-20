from typing import Optional

import torch
from accelerate.utils import get_max_memory
from cloudpathlib import CloudPath
from pydantic import BaseModel, BaseSettings


class Settings(BaseSettings):
    bucket_name: str
    cache_dir: str
    verbose: bool = True

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
            "torch_dtype": torch.float16,
            "variant": "fp16",
        }


class Model(BaseModel):
    variant: str = "fp16"
    name: str = "stabilityai/stable-diffusion-xl-base-1.0"
    vae: str = "madebyollin/sdxl-vae-fp16-fix"
    detector: str = "lllyasviel/Annotators"
    refiner: str = "stabilityai/stable-diffusion-xl-refiner-1.0"
    t2i_adapter: str = "TencentARC/t2i-adapter-lineart-sdxl-1.0"
    inpainter: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    loras: dict[str, dict[str, str]] = {
        "base": {name: "sd_xl_offset_example-lora_1.0.safetensors"},
        "inpainter": {"128461": "civitai"},  # Perfect Eyes XL
    }

    resolution: int = 1024


class Params(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    dtype: torch.dtype = torch.float16
    model: Model = Model()
    batch_size: int = 4

    negative_colors = [
        "purple----",
        "pink----",
        "green----",
        "brown----",
        "(low contrast)---",
    ]
    negative_prompt = (
        "extra digit, cropped, worst quality, low quality, fuzzy++, eyes closed"
    )
    # prompt_template = "closeup (4k photo)+ of a ({ethnicity})-- ({gender})-, ({prompt})++, (cinematic camera)+, highly detailed, (ultra realistic)+, vibrant colors, high contrast, textured skin, realistic dull skin noise, visible skin detail, skin fuzz, dry skin"
    prompt_template = "{prompt}"
    inpaint_prompt_template = "{color} eyes, perfecteyes++, (detailed pupils)+, subtle eyes, natural eyes, realistic eyes, ({ethnicity} {gender})0.1, ({prompt})0.8"
    prompts = [
        "a {gender} dressed as a clown, thin rainbow stripes, suspenders, red nose, (face makeup)--",
        "mysterious, cyberpunk, the universe, cosmos and nebula on clothing",
        "90s style, leather jacket, smug, vintage, antique car, smoking cigar",
        "classy {gender}, wearing a rainbow suit, pop art style, painting by andy warhol",
        "zombie, (decaying skin and clothing)-, (rotting skin)-, inside an abandoned building",
        "(8-bit video game)++, pixelated++, minecraft, lego, blocky, elementary colors"
        "Marvel++ superhero+, superhero costume+, flying in the air, sky+, nyc skyline in background, high contrast, simple colors",
        "a monarch, game of thrones, on the iron throne, wearing a crown+++, magestic, regal, powerful, bold",
        "rock star, face makeup, wearing a slick outfit, performing for fans, grungy, dark colors, moody",
        "character from tron, neon, techno, futuristic, dark background, black clothing, (high contrast)++",
    ]

    seed: Optional[int] = None
    steps: int = 30
    inpainting_steps = 15
    images: int = 4

    detect_resolution: int = 384
    guidance_scale: float = 8.5
    refiner_strength = 0.05
    inpainting_strength = 0.35
    conditioning_strength: tuple[float, float] = (1.50, 1.52)
    conditioning_factor: float = 1.0
    lora_scale = 0.4
    high_noise_frac: float = 1.0
    mask_padding: float = 0.04
