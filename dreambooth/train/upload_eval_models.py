import tempfile

from cloudpathlib import CloudPath
from torch.hub import download_url_to_file

from dreambooth.params import HyperParams
from dreambooth.train.utils import get_model, hash_bytes

BUCKET = "s3://rootvc-photobooth"

URLS = {
    "real_esrgan": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
    "codeformer": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
    "face_detector": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/mmod_human_face_detector-4cb19393.dat",
    "shape_predictor_5": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/shape_predictor_5_face_landmarks-c4b1e980.dat",
}


def main():
    params = HyperParams()
    path = CloudPath(BUCKET) / "models" / params.eval_model_path

    for name, url in URLS.items():
        with tempfile.NamedTemporaryFile() as f:
            download_url_to_file(url, f.name)
            (path / f"{name}.pth").upload_from(f.name)


if __name__ == "__main__":
    main()
