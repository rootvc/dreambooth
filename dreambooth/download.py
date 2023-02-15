from diffusers import StableDiffusionPipeline

from dreambooth.params import HyperParams


def download_model():
    params = HyperParams()
    return StableDiffusionPipeline.from_pretrained(
        params.model.name,
        revision=params.model.revision,
        torch_dtype=params.dtype,
    )


if __name__ == "__main__":
    download_model()