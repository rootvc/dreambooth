import torch
from cloudpathlib import CloudPath
from pydantic import BaseModel, BaseSettings


class Settings(BaseSettings):
    bucket_name: str
    cache_dir: str

    @property
    def bucket(self) -> CloudPath:
        return CloudPath(self.bucket_name)


class Model(BaseModel):
    name: str = "stabilityai/stable-diffusion-xl-base-1.0"
    detector: str = "lllyasviel/Annotators"
    refiner: str = "stabilityai/stable-diffusion-xl-refiner-1.0"
    t2i_adapter: str = "TencentARC/t2i-adapter-lineart-sdxl-1.0"
    loras: dict[str, str] = {name: "sd_xl_offset_example-lora_1.0.safetensors"}
    resolution: int = 1024


class Params(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    dtype: torch.dtype = torch.bfloat16
    model: Model = Model()
    batch_size: int = 4

    negative_prompt: str = ", ".join(
        [
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

    prompt_template = '("a closeup portrait photo of a {race} {gender}", "{prompt}", "4k photo, highly detailed, vibrant colors").and()'
    prompts = [
        "a clown in full makeup",
        "a 3D render of a robotic cyborg",
        "a zombie",
        "a vampire",
        "a character from a van Gough painting",
        "an anime character from Naruto",
        "a Marvel superhero",
    ]

    steps: int = 30
    images: int = 1

    detect_resolution: int = 384
    guidance_scale: float = 10.5
    conditioning_strength: float = 0.9
    conditioning_factor: float = 0.9
    lora_scale = 0.4
    high_noise_frac: float = 0.8
    mask_padding: float = 0.05
