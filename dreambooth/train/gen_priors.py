from cloudpathlib import CloudPath

from dreambooth.train.sdxl.train import get_model
from dreambooth.train.shared import hash_bytes

BUCKET = "s3://rootvc-photobooth"


def main():
    model = get_model(instance_images=[b""])
    priors = model.generate_priors()
    path = CloudPath(BUCKET) / "priors" / hash_bytes(priors.prompt.encode())
    try:
        path.rmtree()
    except FileNotFoundError:
        pass
    path.upload_from(priors.data)


if __name__ == "__main__":
    main()
