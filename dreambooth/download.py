from typing import Type

from diffusers import StableDiffusionPipeline

from dreambooth.params import HyperParams


def download(klass: Type, name: str, **kwargs):
    return klass.from_pretrained(name, **kwargs)


def download_model():
    params = HyperParams()
    return download(
        StableDiffusionPipeline,
        params.model.name,
        revision=params.model.revision,
        torch_dtype=params.dtype,
    )


if __name__ == "__main__":
    download_model()
