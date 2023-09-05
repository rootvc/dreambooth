from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import BaseModel


class Model(BaseModel):
    source: Literal["hf", "civitai"] = "hf"
    name: Union[str, Path] = "Lykon/dreamshaper-8"
    refiner: Optional[Union[str, Path]] = "stabilityai/stable-diffusion-xl-refiner-1.0"
    variant: Optional[str] = None
    vae: Optional[Union[str, Path]] = None
    control_net: Optional[Union[str, Path]] = "lllyasviel/control_v11p_sd15_canny"
    resolution: int = 512
    revision: Optional[str] = None
