from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import BaseModel


class Model(BaseModel):
    source: Literal["hf", "civitai"] = "hf"
    name: Union[str, Path] = "stabilityai/stable-diffusion-xl-base-1.0"
    refiner: Optional[Union[str, Path]] = "stabilityai/stable-diffusion-xl-refiner-1.0"
    variant: Optional[str] = None
    vae: Optional[Union[str, Path]] = "madebyollin/sdxl-vae-fp16-fix"
    control_net: Optional[Union[str, Path]] = "diffusers/controlnet-canny-sdxl-1.0"
    resolution: int = 768
    revision: Optional[str] = None
