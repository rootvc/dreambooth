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


class IPAdapterFiles(BaseModel):
    adapter: str
    image_encoder: str
    image_encoder_config: str


class IPAdapter(BaseModel):
    repo: str
    files: IPAdapterFiles
    vae: str
    base: str


class Model(BaseModel):
    variant: str = "fp16"
    name: str = "stabilityai/stable-diffusion-xl-base-1.0"
    vae: str = "madebyollin/sdxl-vae-fp16-fix"
    refiner: str = "stabilityai/stable-diffusion-xl-refiner-1.0"
    controlnet: str = "diffusers/controlnet-canny-sdxl-1.0"
    ip_adapter: IPAdapter = IPAdapter(
        repo="h94/IP-Adapter",
        files=IPAdapterFiles(
            adapter="models/ip-adapter-full-face_sd15.safetensors",
            image_encoder="models/image_encoder/model.safetensors",
            image_encoder_config="models/image_encoder/config.json",
        ),
        vae="stabilityai/sd-vae-ft-mse",
        base="SG161222/Realistic_Vision_V4.0_noVAE",
    )
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
    background: PromptStrings | None = None
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
            negatives=[P("cross eyed", 3), "eyes closed", "extra eyes"],
        ),
        merge=PromptStrings(
            positives=[
                "realistic",
                "{prompt}",
                "4k",
                "cohesive",
                "detailed",
                "high quality",
            ],
            negatives=["blurry", "fuzzy", "disjointed", "low quality", "lowres"],
        ),
        details=PromptStrings(
            positives=["4k photo", "{prompt}", "closeup portrait"],
            negatives=[
                "monochrome",
                "lowres",
                "bad anatomy",
                "worst quality",
                "low quality",
            ],
        ),
    )

    prompts = [
        "a clown on a sunny day, thin rainbow stripe suspenders",
        "mysterious, floating in the universe, cosmos reflected in clothing, cyberpunk vibes",
        "90s style, leather jacket, smug, vintage, holding a smoking cigar",
        "classy, pinstripe suit, pop art style, andy warhol",
        # "zombie, decaying skin, torn clothing, inside an abandoned building",
        # "(person in a mario costume)+, super mario, pixelated, (elementary colors)-",
        # "Marvel superhero, sky in the background, comic book style",
        # "a monarch wearing a crown, game of thrones, on the iron throne, magestic, regal, powerful, bold",
        # "character from tron, neon, techno, futuristic, dark background, black clothing, high contrast",
        # "sassy yearbook photo, high school, teenage angst, creative",
        # "a hero from lord of the rings, fantasy, medieval, countryside",
        # "a student from harry potter, magic, fantasy",
        # "a robot come to life, industrial, metal, wires",
        # "a politician at a podium, presidential, confident, powerful",
        # "darth vader, star wars, dark side, powerful, evil",
    ]

    seed: Optional[int] = None
    steps: int = 25
    inpainting_steps = 15
    images: int = 1

    detect_resolution: int = 384
    guidance_scale: float = 12.5
    refine_guidance_scale: float = 7.5
    refiner_strength = 0.75
    inpainting_strength = 0.40
    conditioning_strength: tuple[float, float] = (0.70, 0.75)
    conditioning_factor: float = 0.75
    lora_scale = 0.4
    mask_padding: float = 0.055
