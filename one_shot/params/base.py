from typing import Optional

import torch
from pydantic import BaseModel

from one_shot.params.prompts import PromptTemplates


class Model(BaseModel):
    variant: str = "fp16"
    name: str = "stabilityai/stable-diffusion-xl-base-1.0"
    vae: str = "madebyollin/sdxl-vae-fp16-fix"
    refiner: str = "stabilityai/stable-diffusion-xl-refiner-1.0"
    inpainter: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    vqa: str = "ybelkada/blip-vqa-base"
    sam: str = "facebook/sam-vit-huge"
    loras: dict[str, dict[str, str]] = {
        "base": {
            name: "sd_xl_offset_example-lora_1.0.safetensors",
            "131991": "civitai",  # Juggernaut Cinematic XL
        },
        "inpainter": {
            name: "sd_xl_offset_example-lora_1.0.safetensors",
            "128461": "civitai",  # Perfect Eyes XL
            "131991": "civitai",  # Juggernaut Cinematic XL
            "152685": "civitai",  # Wallpaper X
            "156002": "civitai",  # PAseer-SDXL-Weird DreamLand
        },
    }
    resolution: int = 1024


class Params(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    dtype: torch.dtype = torch.float16
    model: Model = Model()
    batch_size: int = 4
    prompt_templates: PromptTemplates
    prompts: list[str] = [
        "a clown on a sunny day, thin rainbow stripe suspenders",
        "mysterious, floating in the universe, cosmos reflected in clothing, cyberpunk vibes",
        "90s style, leather jacket, smug, vintage, holding a smoking cigar",
        "classy, pinstripe suit, pop art style, andy warhol",
        "zombie, decaying skin, torn clothing, inside an abandoned building",
        "(person in a mario costume)+, super mario, pixelated, (elementary colors)-",
        "Marvel superhero, sky in the background, comic book style",
        "a monarch wearing a crown, game of thrones, on the iron throne, magestic, regal, powerful, bold",
        "character from tron, neon, techno, futuristic, dark background, black clothing, high contrast",
        "sassy yearbook photo, high school, teenage angst, creative",
        "a hero from lord of the rings, fantasy, medieval, countryside",
        "a student from harry potter, magic, fantasy",
        "a robot come to life, industrial, metal, wires",
        "a politician at a podium, presidential, confident, powerful",
        "darth vader, star wars, dark side, powerful, evil",
    ]

    seed: Optional[int] = None
    steps: int = 25
    inpainting_steps: int = 15
    images: int = 4

    detect_resolution: int = 384
    lora_scale: float = 0.4
    mask_padding: float = 0.055

    guidance_scale: float
    refine_guidance_scale: float
    refiner_strength: float
    inpainting_strength: float
    conditioning_strength: tuple[float, float]
    conditioning_factor: float
