import os
import tempfile
from pathlib import Path

from cloudpathlib import CloudPath
from torch.hub import download_url_to_file

from dreambooth.params import HyperParams

BUCKET = "s3://rootvc-photobooth"

URLS = {
    "realesrgan": [
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    ],
    "CodeFormer": [
        "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
    ],
    "facexlib": [
        "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
        "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
    ],
}


def upload():
    params = HyperParams()
    path = CloudPath(BUCKET) / "models" / params.eval_model_path

    for name, url in [(name, url) for name, urls in URLS.items() for url in urls]:
        with tempfile.NamedTemporaryFile() as f:
            download_url_to_file(url, f.name)
            (path / name / Path(url).name).upload_from(f.name)


def main():
    model_dir = Path(os.environ["HF_MODEL_CACHE"])
    models = model_dir / HyperParams().eval_model_path

    for name, url in [(name, url) for name, urls in URLS.items() for url in urls]:
        path = models / name / Path(url).name
        path.parent.mkdir(parents=True, exist_ok=True)
        download_url_to_file(url, path)


if __name__ == "__main__":
    main()
