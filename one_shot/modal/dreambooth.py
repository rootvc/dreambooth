import random
from pathlib import Path
from tempfile import TemporaryDirectory

from modal import method, web_endpoint

from one_shot.dreambooth import OneShotDreambooth
from one_shot.dreambooth.request import Request
from one_shot.modal import fn_kwargs, stub, volume


class _Dreambooth(OneShotDreambooth):
    def __init__(self):
        super().__init__(volume)

    def _generate(self, id: str):
        image = Request(self, id).generate()
        with TemporaryDirectory() as dir:
            file = Path(dir) / "grid.png"
            image.save(file)
            (self.settings.bucket / "output" / id / "grid.png").upload_from(
                file, force_overwrite_to_cloud=True
            )

            image.save(file.with_suffix(".jpg"), optimize=True)
            (self.settings.bucket / "output" / id / "grid.jpg").upload_from(
                file.with_suffix(".jpg"), force_overwrite_to_cloud=True
            )


@stub.cls(**fn_kwargs)
class Dreambooth(_Dreambooth):
    @method()
    def tune(self, ids: list[str] = ["test"], params: dict = {}):
        random.shuffle(ids)
        for id in ids:
            yield from Request(self, id).tune(params)

    @method()
    def generate(self, id: str):
        return self._generate(id)

    @web_endpoint(custom_domains=["warm.modal.root.vc"], method="POST")
    def warm(self):
        pass


@stub.cls(**fn_kwargs, keep_warm=1)
class PersistentDreambooth(_Dreambooth):
    @web_endpoint(
        wait_for_response=False, custom_domains=["dream.modal.root.vc"], method="POST"
    )
    def dream(self, id: str):
        return self._generate(id)
