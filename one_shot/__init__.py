from pathlib import Path
from tempfile import mkdtemp

from loguru import logger

from one_shot.config import init_config, init_logging
from one_shot.dreambooth import OneShotDreambooth
from one_shot.modal import fn_kwargs, stub, volume
from one_shot.modal.dreambooth import Dreambooth

logger.info("Initializing config...")
init_logging(logger)
init_config()


@stub.local_entrypoint()
def main():
    dir = Path(mkdtemp())
    logger.info(f"Using {dir} as cache directory.")

    for i, img in enumerate(
        Dreambooth().tune.remote_gen(
            [
                # "test",
                # "4d25928d3214ec2a4fab8e39678948ee",
                # "a27209d1579ee29baaab83d0af4a28e9",
                # "b609042250a49a12edf7b27541c19c89",
                # "bc8f34a2c5e6b3eb05f77cb3235ae440",
                # "e4e7b1cd9b93a20c027645e57d9c069f",
                # "ea829f9d41697757a51dfe49842652c6",
                # "f09e2b714736a0a553d33448fd6d9ed5",
                # "f314e6cb3429719e3bdca6ac28823ccc",
                "ca4b1e40984e7cc6f23777963e9ae76e",
            ],
            {
                "adapter_conditioning_scale": [1.65, 1.7, 1.75, 2.28, 2.35],
                "refiner_strength": [0.01, 0.05, 0.09, 0.30],
                "guidance_scale": [14, 16.5, 18, 10],
                "high_noise_frac": [0.90, 0.95],
            },
        )
    ):
        img.save(dir / f"{i}.png")


@stub.function(**fn_kwargs)
def seed():
    dreambooth = OneShotDreambooth(volume)
    dreambooth.__enter__(skip_procs=True)
    dreambooth.__exit__(None, None, None)
