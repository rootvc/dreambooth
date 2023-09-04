from cloudpathlib import CloudPath

from dreambooth.train.base import get_params
from dreambooth.train.sdxl.train import get_model
from dreambooth.train.shared import dprint, hash_bytes

BUCKET = "s3://rootvc-photobooth"


def main():
    params = get_params()
    params.model.name = "stabilityai/stable-diffusion-xl-base-1.0"
    params.model.vae = "madebyollin/sdxl-vae-fp16-fix"
    params.model.revision = None
    params.model.resolution = 1024
    params.batch_size = 10

    model = get_model(instance_images=[b""], params=params)
    priors = model.generate_priors()

    dprint("Uploading...")

    path = CloudPath(BUCKET) / "priors" / hash_bytes(priors.prompt.encode())
    try:
        path.rmtree()
    except Exception:
        pass
    path.upload_from(priors.data)


if __name__ == "__main__":
    main()
