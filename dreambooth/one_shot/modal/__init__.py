from modal import Image as DockerImage
from modal import Secret, Stub, Volume, gpu

stub = Stub()
volume = stub.volume = Volume.persisted("model-cache")

fn_kwargs = {
    "image": DockerImage.from_registry("rootventures/train-dreambooth-modal:latest")
    .pip_install("loguru")
    .apt_install("lsof")
    .dockerfile_commands(["RUN python -m pip install --no-deps retina-face"]),
    "gpu": gpu.A100(count=2),
    "memory": 30518,
    "cpu": 8.0,
    "volumes": {"/root/cache": volume},
    "secret": Secret.from_name("dreambooth"),
}
