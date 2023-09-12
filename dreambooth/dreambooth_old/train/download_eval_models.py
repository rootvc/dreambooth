import logging
import os
import tempfile
from pathlib import Path

from cloudpathlib import CloudPath
from torch.hub import download_url_to_file

from dreambooth_old.params import HyperParams

logger = logging.getLogger(__name__)

BUCKET = "s3://rootvc-photobooth"

URLS = {
    "mediapipe": [
        "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
    ],
}


def upload():
    params = HyperParams()
    path = CloudPath(BUCKET) / "models" / params.eval_model_path

    for name, url in [(name, url) for name, urls in URLS.items() for url in urls]:
        with tempfile.NamedTemporaryFile() as f:
            download_url_to_file(url, f.name)
            (path / name / Path(url).name).upload_from(f.name)


def download(root: Path):
    downloaded = False
    for name, url in [(name, url) for name, urls in URLS.items() for url in urls]:
        path = root / name / Path(url).name
        if path.exists():
            continue
        else:
            downloaded = True
        logger.info(f"Downloading {url} to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        download_url_to_file(url, path)
    return downloaded


def main():
    model_dir = Path(os.environ["HF_MODEL_CACHE"])
    models = model_dir / HyperParams().eval_model_path
    download(models)


if __name__ == "__main__":
    main()
