from modal import Image as DockerImage
from modal import Stub, Volume, method

from one_shot.dreambooth import OneShotDreambooth, Request

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

    @method()
    def warm(self):
        return Request(self, "test").generate()

    @method()
    def generate(self, id: str):
        return Request(self, id).generate()


@stub.local_entrypoint()
def main():
    Dreambooth().warm.remote()
