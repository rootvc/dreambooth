from modal import method

from one_shot.config import init_config

init_config(split_gpus=True)

from one_shot.dreambooth import OneShotDreambooth, Request
from one_shot.modal import fn_kwargs, stub, volume


@stub.cls(**fn_kwargs)
class Dreambooth(OneShotDreambooth):
    def __init__(self):
        super().__init__(volume)

    @method()
    def warm(self):
        return Request(self, "test").generate()

    @method()
    def generate(self, id: str):
        return Request(self, id).generate()
