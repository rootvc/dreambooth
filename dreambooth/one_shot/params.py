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
        "inpainter": {"118427": "civitai"},
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
            "(eyes closed)++",
            "(deformed eyes)++",
            "crossed eyes",
            "deformed iris",
            "out of focus",
            "lens blur",
            "extra digit",
            "fewer digits",
            "cropped",
            "worst quality",
            "low quality",
            "glitch",
            "deformed",
            "mutated",
            "ugly",
            "disfigured",
        ]
    )

    inpaint_prompt_template = "({race} {gender})---, {color} eyes, perfecteyes"
    prompt_template = "a closeup portrait photo of a ({race})- {gender}, ({prompt})++++, 4k photo, highly detailed, (vibrant colors)+"
    prompts = [
        "a clown in full makeup",
        "a 3D render of a robotic cyborg",
        "a zombie",
        "a vampire",
        "a character from a van Gough painting",
        "an anime character from Naruto",
        "a Marvel superhero",
    ]

    steps: int = 25
    images: int = 4

    detect_resolution: int = 384
    guidance_scale: float = 4.5
    inpainting_strength = 0.5
    conditioning_strength: float = 1.3
    conditioning_factor: float = 1.0
    lora_scale = 0.4
    high_noise_frac: float = 0.8
    mask_padding: float = 0.05
