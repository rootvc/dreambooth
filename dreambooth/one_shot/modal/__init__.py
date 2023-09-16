from modal import Image as DockerImage
from modal import Secret, Stub, Volume, gpu

stub = Stub()
volume = stub.volume = Volume.persisted("model-cache")

fn_kwargs = {
    "image": DockerImage.from_registry("rootventures/train-dreambooth-modal:latest"),
    "gpu": gpu.A100(count=3),
    "memory": 30518,
    "cpu": 8.0,
    "volumes": {"/root/cache": volume},
    "secret": Secret.from_name("dreambooth"),
    "timeout": 60 * 15,
}
