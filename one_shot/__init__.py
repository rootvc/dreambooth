from modal import Image as DockerImage
from modal import Stub, Volume

from .dreambooth import OneShotDreambooth

stub = Stub()
volume = stub.volume = Volume.persisted("model-cache")


@stub.cls(
    image=DockerImage.from_registry("rootventures/train-dreambooth-modal:latest"),
    gpu="A100",
    volumes={"/root/cache": volume},
)
class Dreambooth(OneShotDreambooth):
    def __init__(self):
        super().__init__(volume)
