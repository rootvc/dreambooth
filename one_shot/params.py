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


class PromptSegment(BaseModel):
    raw: str
    modifier: float

    @classmethod
    def new(cls, raw: str, modifier: float):
        return cls(raw=raw, modifier=modifier)

    @classmethod
    def plus(cls, raw: str, count: int = 1):
        return cls(raw=raw, modifier=1.1**count)

    @classmethod
    def minus(cls, raw: str, count: int = 1):
        return cls(raw=raw, modifier=0.9**count)

    def __str__(self):
        if self.modifier:
            return f"({self.raw}){self.modifier:.2f}"
        else:
            return self.raw


class PromptStrings(BaseModel):
    positives: list[str | PromptSegment]
    negatives: list[str | PromptSegment]

    def positive(self, **kwargs) -> str:
        return ", ".join(map(str, self.positives)).format(**kwargs)

    def negative(self, **kwargs) -> str:
        return ", ".join(map(str, self.negatives)).format(**kwargs)


class PromptTemplates(BaseModel):
    background: PromptStrings
    eyes: PromptStrings
    merge: PromptStrings
    details: PromptStrings


F = PromptSegment.new
P = PromptSegment.plus
M = PromptSegment.minus


class Params(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    dtype: torch.dtype = torch.float16
    model: Model = Model()
    batch_size: int = 4

    prompt_templates: PromptTemplates = PromptTemplates(
        background=PromptStrings(
            positives=[
                "closeup portrait",
                "large head",
                "{prompt}",
                "{ethnicity} {gender}",
                "4k",
                "realistic",
                "cinematic",
                "cinematic effect",
                P("hyperrealistic"),
                P("contrasts", 2),
                "sharp",
                P("highly detailed"),
            ],
            negatives=["cropped", "worst quality", "low quality"],
        ),
        eyes=PromptStrings(
            positives=[
                "{color} eyes",
                P("perfecteyes", 2),
                P("detailed pupils"),
                "subtle eyes",
                "natural eyes",
                "realistic eyes",
                F("{ethnicity} {gender}", 0.1),
                F("{prompt}", 0.8),
            ],
            negatives=["eyes closed", "extra eyes"],
        ),
        merge=PromptStrings(
            positives=["4k", "cohesive", "detailed", F("{prompt}", 0.2)],
            negatives=["blurry", "fuzzy", "disjointed", "eyes"],
        ),
        details=PromptStrings(
            positives=[
                "{prompt}",
                "{ethnicity} {gender}",
                "4k",
                "realistic",
                "cinematic",
                "cinematic effect",
                P("hyperrealistic"),
                P("contrasts", 2),
                "sharp",
                P("highly detailed"),
                "white teeth",
                "nice smile",
                F("airbrushed", 0.2),
                F("beautiful", 0.2),
                "dream",
            ],
            negatives=[
                P("extra fingers", 2),
                P("ugly", 2),
                P("blurry", 3),
                P("incomplete", 2),
                P("fuzzy"),
                "large head",
                F("picture frame", 1.5),
                "disjointed",
                "monotone",
                "dreary",
                "extra digit",
                "eyes closed",
                "extra eyes",
                "bad smile",
                M("cropped"),
                M("worst quality"),
                M("low quality"),
                "glitch",
                "deformed",
                "mutated",
                "disfigured",
                "yellow teeth",
            ],
        ),
    )

    prompts = [
        "a clown on a sunny day, thin rainbow stripe suspenders",
        "mysterious, floating in the universe, cosmos and nebula reflected in clothing, cyberpunk vibes",
        "90s style, leather jacket, smug, vintage, smoking cigar",
        "classy, pinstripe suit, pop art style, andy warhol",
        "zombie, decaying skin, torn clothing, inside an abandoned building",
        "(character from mario)+, super mario, pixelated, (elementary colors)-"
        "Marvel superhero, sky in the background, comic book style",
        "a monarch wearing a crown, game of thrones, on the iron throne, magestic, regal, powerful, bold",
        "character from tron, neon, techno, futuristic, dark background, black clothing, high contrast",
        "sassy yearbook photo, high school, teenage angst, creative",
    ]

    seed: Optional[int] = None
    steps: int = 25
    inpainting_steps = 15
    images: int = 4

    detect_resolution: int = 384
    guidance_scale: float = 9.0
    refiner_strength = 0.05
    inpainting_strength = 0.40
    conditioning_strength: tuple[float, float] = (1.8, 1.9)
    conditioning_factor: float = 1.0
    lora_scale = 0.4
    high_noise_frac: float = 0.85
    mask_padding: float = 0.055
