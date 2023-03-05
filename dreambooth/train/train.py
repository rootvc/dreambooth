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
    model_dir = tempfile.mkdtemp()

    if not env.is_main:
        return model_dir

    model_file = model_data / Path(name).with_suffix(".tpxz")
    with tempfile.NamedTemporaryFile() as f:
        subprocess.check_call(
            [
                "pixz",
                "-d",
                model_file,
                f.name,
            ]
        )
        shutil.unpack_archive(f.name, model_dir, format="tar")
    return model_dir


def _unpack_eval_models(env: environment.Environment):
    params = get_params()
    models = Path(env.channel_input_dirs["model"]) / params.eval_model_path
    os.symlink(models, "./weights", target_is_directory=True)


def sagemaker_params(env: environment.Environment) -> Params:
    train_data = Path(env.channel_input_dirs["train"])
    prior_data = Path(env.channel_input_dirs["prior"])
    cache_data = Path(env.channel_input_dirs["cache"])
    params = get_params()

    params.model.name = _unpack_model(env, params.model.name)
    if params.model.vae:
        params.model.vae = _unpack_model(env, params.model.vae)
    params.prior_class = Class(prompt_=params.prior_prompt, data=prior_data)

    if env.is_main:
        shutil.copytree(cache_data, os.environ["CACHE_DIR"], dirs_exist_ok=True)
        _unpack_eval_models(env)

    return {"instance_path": train_data, "params": params}


def standalone_params(env: environment.Environment):
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
        model.train()
    finally:
        model.accelerator.wait_for_everyone()
        if env.is_main and env.channel_input_dirs:
            shutil.copytree(model.output_dir, env.model_dir, dirs_exist_ok=True)
            shutil.copytree(
                os.environ["CACHE_DIR"],
                env.channel_input_dirs["cache"],
                dirs_exist_ok=True,
            )


if __name__ == "__main__":
    main()
