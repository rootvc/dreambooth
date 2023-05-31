import os
import traceback
import warnings

import runpod.serverless

from dreambooth.train.shared import Mode
from dreambooth.train.train import get_model, standalone_params
from dreambooth.train.train import main as train

PARTY_MODE = os.environ.get("PARTY_MODE", "0") == "1"


def prepare():
    print("Warming...")

    os.environ["WARM"] = "1"
    os.environ["DREAMBOOTH_ID"] = "test"
    if PARTY_MODE:
        os.environ["WANDB_MODE"] = "disabled"

    try:
        if PARTY_MODE:
            train()
        else:
            params = standalone_params(True)
            model = get_model(**params)
            dataset = model._prepare_dataset()
            model._prepare_models(model._load_models(dataset, Mode.TI), Mode.TI)
    finally:
        if PARTY_MODE:
            del os.environ["WANDB_MODE"]
        del os.environ["DREAMBOOTH_ID"]
        del os.environ["WARM"]


def run(job):
    print("Running job", job)
    for k, v in job["input"]["env"].items():
        os.environ[k] = v
    try:
        train()
        return {"status": "ok"}
    finally:
        for k in job["input"]["env"]:
            del os.environ[k]


def main():
    try:
        prepare()
        pass
    except Exception:
        traceback.print_exc()
        warnings.warn("Warmup failed. Requests will likely fail too.")
    runpod.serverless.start({"handler": run})


if __name__ == "__main__":
    main()
