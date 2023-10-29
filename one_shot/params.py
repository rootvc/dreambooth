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
    vqa: str = "ybelkada/blip-vqa-base"
    sam: str = "facebook/sam-vit-huge"
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

    negative_prompt = "boxy, rectangle+, extra fingers, ugly+, blurry+, fuzzy+, monotone, dreary, extra digit, fewer digits, eyes closed, extra eyes, bad smile, cropped, worst quality, low quality, glitch, deformed, mutated, disfigured, text"
    refine_negative_prompt = "(double face)++, (extra eyes)+, (multiple faces)+, extra digits, extra fingers, deformed"
    prompt_template = "{prompt}, {ethnicity} {gender}, 4k photo, cinematic effect, hyperrealistic+, sharp, (highly detailed)+, (airbrushed)0.2, (beautiful)0.2"
    inpaint_prompt_template = "{color} eyes, perfecteyes++, (detailed pupils)+, subtle eyes, natural eyes, realistic eyes, ({ethnicity} {gender})0.1, ({prompt})0.8"
    prompts = [
        "a {ethnicity}0.3 {gender} dressed as a clown, goofy, thin rainbow stripes, suspenders, red button nose",
        "mysterious {ethnicity}0.3 {gender}, cyberpunk, the universe, cosmos and nebula on clothing, powerful",
        # "90s style, leather jacket, smug, vintage, antique car, smoking cigar",
        # "classy {gender}, wearing a pinstripe suit, pop art style, painting by andy warhol",
        # "zombie, (decaying skin and clothing)-, (rotting skin)-, inside an abandoned building",
        # "(8-bit video game)++++, pixelated++, minecraft+, lego, blocky, elementary colors"
        # "Marvel++ superhero+, superhero costume+, mask, simple color, flying in the air, high contrast",
        # "a monarch+ wearing a crown+++, game of thrones, on the iron throne, magestic, regal, powerful, bold",
        # "character from tron, neon, techno, futuristic, dark background, black clothing, (high contrast)++",
    ] * 2

    seed: Optional[int] = None
    steps: int = 25
    inpainting_steps = 20
    images: int = 2

    detect_resolution: int = 384
    guidance_scale: float = 9.0
    refiner_strength = 0.05
    inpainting_strength = 0.40
    conditioning_strength: tuple[float, float] = (1.50, 1.52)
    conditioning_factor: float = 1.0
    lora_scale = 0.4
    high_noise_frac: float = 0.80
    mask_padding: float = 0.05
