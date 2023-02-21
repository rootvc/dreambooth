from cloudpathlib import CloudPath

from dreambooth.train.utils import get_model, hash_bytes

BUCKET = "s3://rootvc-photobooth"


def main():
    model = get_model(instance_images=[b""])
    priors = model.generate_priors(progress_bar=True)
    path = CloudPath(BUCKET) / "priors" / hash_bytes(priors.prompt.encode())
    path.upload_from(priors.data)


if __name__ == "__main__":
    main()
