import tempfile

from cloudpathlib import CloudPath
from torch.hub import download_url_to_file

from dreambooth.params import HyperParams
from dreambooth.train.utils import get_model, hash_bytes

BUCKET = "s3://rootvc-photobooth"

REAL_ESRGAN_URL = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth"


def main():
    params = HyperParams()
    path = CloudPath(BUCKET) / "models"

    with tempfile.NamedTemporaryFile() as f:
        download_url_to_file(REAL_ESRGAN_URL, f.name)
        (path / params.real_esrgan_path).upload_from(f.name)


if __name__ == "__main__":
    main()
