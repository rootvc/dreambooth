from modal import Image as DockerImage
from modal import Secret, Stub, Volume, gpu

from one_shot.params import Settings

settings = Settings()

stub = Stub("dreambooth-one-shot")
volume = stub.volume = Volume.persisted("model-cache")

fn_kwargs = {
    "image": DockerImage.from_registry("rootventures/train-dreambooth-modal:latest")
    .pip_install("snoop", "scikit-image")
    .env({"TORCH_HOME": "/root/cache/torch"}),
    "gpu": gpu.A100(count=1),
    "memory": 22888,
    "cpu": 4.0,
    "volumes": {"/root/cache": volume},
    "secret": Secret.from_name("dreambooth"),
    "timeout": 60 * 30,
    "cloud": "gcp",
    # "container_idle_timeout": 60 * 10,
}
