import os
from pathlib import Path
from typing import Type

from diffusers import AutoencoderKL, StableDiffusionPipeline
from transformers import (
    CLIPModel,
    CLIPProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from dreambooth.params import HyperParams

HF_MODEL_CACHE = os.getenv("HF_MODEL_CACHE")


def download(klass: Type, name: str, **kwargs):
    model = klass.from_pretrained(name, **kwargs)
    if HF_MODEL_CACHE:
        path = Path(HF_MODEL_CACHE) / name
        print(f"Saving {name} to {path}")
        model.save_pretrained(path)
    return model


def download_test_models(_, name: str):
    return [
        download(klass, name)
        for klass in [
            CLIPProcessor,
            CLIPTextModelWithProjection,
            CLIPTokenizer,
            CLIPVisionModelWithProjection,
            CLIPModel,
        ]
    ]


def download_model():
    params = HyperParams()
    models = [
        download(
            StableDiffusionPipeline,
            params.model.name,
            revision=params.model.revision,
            torch_dtype=params.dtype,
        )
    ]
    if params.model.vae:
        models.append(download(AutoencoderKL, params.model.vae))
    return models


if __name__ == "__main__":
    download_model()
    download_test_models(None, HyperParams().test_model)
