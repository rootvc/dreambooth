import json
import os
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import TypedDict

from sagemaker_training import environment

from dreambooth.params import Class
from dreambooth.train.utils import HyperParams, get_model, get_params

IGNORE_MODS = ["_functorch", "fmha", "torchvision", "tempfile"]
IGNORE_RE = r"|".join([rf"(.*)\.{mod}\.(.*)" for mod in IGNORE_MODS])
warnings.filterwarnings("ignore", module=IGNORE_RE)
for klass in [ImportWarning, DeprecationWarning, ResourceWarning]:
    warnings.filterwarnings("ignore", category=klass)


class Params(TypedDict):
    instance_path: Path
    params: HyperParams


def _unpack_model(env: environment.Environment, name: str):
    model_data = Path(env.channel_input_dirs["model"])
    model_dir = Path("models") / name

    if not env.is_main:
        return model_dir

    os.makedirs(model_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile() as f:
        subprocess.check_call(
            [
                "pixz",
                "-d",
                model_data / Path(name).with_suffix(".tpxz"),
                f.name,
            ]
        )
        shutil.unpack_archive(f.name, model_dir, format="tar")
    return model_dir


def _unpack_eval_models(env: environment.Environment):
    params = get_params()
    models = Path(env.channel_input_dirs["model"]) / params.eval_model_path
    os.symlink(models, "weights", target_is_directory=True)


def _setup_global_cache(env):
    import torch._inductor.config

    torch._inductor.config.global_cache_path = (
        Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]) / "global_cache"
    )


def _persist_global_cache():
    import torch
    import torch.version

    cache_dir = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"])
    local_cache = cache_dir / "local_cache"
    global_cache = cache_dir / "global_cache"

    dinfo = torch.cuda.get_device_properties(torch.cuda.current_device()).name
    vinfo = torch.version.cuda

    local_cache_data = json.loads(local_cache.read_text())
    global_cache_data = {dinfo: {vinfo: local_cache_data}}

    global_cache.write_text(json.dumps(global_cache_data))


def sagemaker_params(env: environment.Environment) -> Params:
    train_data = Path(env.channel_input_dirs["train"])
    prior_data = Path(env.channel_input_dirs["prior"])
    cache_data = Path(env.channel_input_dirs["cache"])
    output_data = Path(env.channel_input_dirs["output"])
    params = get_params()

    params.model.name = _unpack_model(env, params.model.name)
    if params.model.vae:
        params.model.vae = _unpack_model(env, params.model.vae)
    params.prior_class = Class(prompt_=params.prior_prompt, data=prior_data)
    params.image_output_path = output_data
    params.model_output_path = Path(env.model_dir)

    _setup_global_cache(env)
    if env.is_main:
        shutil.copytree(cache_data, os.environ["CACHE_DIR"], dirs_exist_ok=True)
        _unpack_eval_models(env)

    return {"instance_path": train_data, "params": params}


def standalone_params(env: environment.Environment):
    # import boto
    # download input data, and upload output at end (both to S3)
    params = get_params()
    example_data = Path(__file__).parent.parent / "data" / "example"
    return {"instance_path": example_data, "params": params}


def main():
    env = environment.Environment()
    env.is_main = os.getenv("LOCAL_RANK", "-1") == "0"

    if env.channel_input_dirs:
        params = sagemaker_params(env)
    else:
        params = standalone_params(env)

    model = get_model(**params)

    try:
        model.accelerator.wait_for_everyone()
        model.train()
        model.eval()
    except Exception:
        if env.is_main:
            model.accelerator.end_training()
            raise
    finally:
        model.accelerator.wait_for_everyone()
        if env.is_main and env.channel_input_dirs:
            _persist_global_cache()
            shutil.copytree(
                os.environ["CACHE_DIR"],
                env.channel_input_dirs["cache"],
                dirs_exist_ok=True,
            )


if __name__ == "__main__":
    main()
