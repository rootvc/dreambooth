from diffusers import StableDiffusionPipeline
from utils import Trainer, get_params


def download_model():
    params = get_params()
    StableDiffusionPipeline.from_pretrained(
        params.model.name,
        revision=params.model.revision,
        torch_dtype=Trainer.DTYPE,
    )


if __name__ == "__main__":
    download_model()
