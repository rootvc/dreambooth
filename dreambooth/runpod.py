import os
import traceback
import warnings
from collections import namedtuple

import runpod.serverless

from dreambooth.train.eval import Evaluator
from dreambooth.train.test import Tester
from dreambooth.train.train import main as train
from dreambooth.train.train import standalone_params
from dreambooth.train.utils import get_model

Acc = namedtuple("Acc", ["device"])
Class = namedtuple("Acc", ["data"])


def prepare():
    print("Warming...")

    os.environ["WARM"] = "1"
    os.environ["DREAMBOOTH_ID"] = "test"
    os.environ["WANDB_MODE"] = "disabled"

    params = standalone_params(True)
    trainer = get_model(**params)
    pipe = trainer._pipeline()

    tester = Tester(params["params"], Acc(device="cuda"), None)
    evaluator = Evaluator(
        trainer.accelerator.device,
        params["params"],
        Class(data=params["instance_path"]),
        pipe,
    )

    trainer.generate_depth_values(params["instance_path"])
    trainer.generate_depth_values(params["params"].prior_class.data)

    # trainer._prepare_models(trainer._prepare_dataset(), Mode.TI)
    # trainer._prepare_models(trainer._prepare_dataset(), Mode.LORA)
    tester.clip_models()
    evaluator._upsampler()
    evaluator._restorer()

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
    except Exception:
        traceback.print_exc()
        warnings.warn("Warmup failed. Requests will likely fail too.")
    runpod.serverless.start({"handler": run})


if __name__ == "__main__":
    main()
