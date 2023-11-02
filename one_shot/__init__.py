from loguru import logger
from modal import is_local

from one_shot.config import init_config, init_logging
from one_shot.dreambooth import OneShotDreambooth
from one_shot.modal import fn_kwargs, stub, volume
from one_shot.modal.dreambooth import Dreambooth

init_logging(logger)

if not is_local():
    logger.info("Initializing config...")
    init_config()


def _main():
    logger.info("Calling remote function...")
    for i, path in enumerate(
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
                # "ca4b1e40984e7cc6f23777963e9ae76e",
                # "d41d8cd98f00b204e9800998ecf8427e",
                "9806ad626c78b3b4b1547c73cc627605",
                "79c053cf2d6d3922be669f9b78f34b2a",
            ],
            {
                # "steps": [30],
                # "conditioning_strength": [1.50],
                # "conditioning_factor": [1.0],
                # "seed": [42],
                # "refiner_strength": [0.05],
                # "guidance_scale": [8.5],
                "seed": [42],
                # "high_noise_frac": [1.0],
            },
        )
    ):
        logger.info("Saved {}: {}", i, path)


main = stub.local_entrypoint()(_main)


@stub.local_entrypoint()
async def test(id: str = "ca4b1e40984e7cc6f23777963e9ae76e"):
    await Dreambooth().generate.remote.aio(id)


@stub.function(**fn_kwargs)
def seed():
    dreambooth = OneShotDreambooth(volume)
    dreambooth.__enter__(skip_procs=True)
    dreambooth.__exit__(None, None, None)
