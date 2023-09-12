from loguru import logger
from modal import is_local

from one_shot.config import init_config, init_logging
from one_shot.modal import fn_kwargs, stub, volume

if is_local():
    import dreambooth_old  # noqa: F401

    from one_shot.modal.dreambooth import Dreambooth  # noqa: F401


init_logging(logger)


@stub.local_entrypoint()
def main():
    from one_shot.modal.dreambooth import Dreambooth

    Dreambooth().warm.remote().show()


@stub.function(**fn_kwargs, timeout=60 * 15)
def seed():
    init_config(split_gpus=False)
    from one_shot.dreambooth import OneShotDreambooth, Request

    with OneShotDreambooth(volume) as dreambooth:
        Request(dreambooth, "test").face.compile_models()
