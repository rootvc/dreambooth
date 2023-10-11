import random

from modal import method

from one_shot.dreambooth import OneShotDreambooth
from one_shot.dreambooth.request import Request
from one_shot.modal import fn_kwargs, stub, volume


@stub.cls(**fn_kwargs)
class Dreambooth(OneShotDreambooth):
    def __init__(self):
        super().__init__(volume)

    @method()
    def tune(self, ids: list[str] = ["test"], params: dict = {}):
        random.shuffle(ids)
        for id in ids:
            yield from Request(self, id).tune(params)

    @method()
    def generate(self, id: str):
        return Request(self, id).generate()
