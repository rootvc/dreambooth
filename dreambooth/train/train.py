import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import TypedDict

from sagemaker_training import environment

from dreambooth.params import Class
from dreambooth.train.shared import dprint
from dreambooth.train.utils import HyperParams, get_model, get_params, hash_bytes

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
    import facelib.utils.misc

    params = get_params()
    models = Path(env.channel_input_dirs["model"]) / params.eval_model_path
    facelib_path = Path(facelib.utils.misc.ROOT_DIR) / "weights"

    os.symlink(models, "weights", target_is_directory=True)
    shutil.rmtree(facelib_path, ignore_errors=True)
    os.symlink(models, facelib_path, target_is_directory=True)


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

    dprint("Persisting local cache...")
    if not local_cache.exists():
        dprint("No local cache found, skipping...")
        return

    dinfo = torch.cuda.get_device_properties(torch.cuda.current_device()).name
    vinfo = torch.version.cuda

    dprint("Reading local cache...")
    local_cache_data = json.loads(local_cache.read_text())
    if global_cache.exists():
        global_cache_data = json.loads(global_cache.read_text())
    else:
        global_cache_data = {}
    global_cache_data[dinfo][vinfo] = local_cache_data

    dprint("Writing global cache...")
    global_cache.write_text(json.dumps(global_cache_data))

    dprint("Removing local cache...")
    local_cache.unlink()


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


def sagemaker_cleanup(env: environment.Environment):
    # No need to persist models
    shutil.rmtree(env.model_dir, ignore_errors=True)
    os.mkdir(env.model_dir)

    dprint("Persisting global cache...")
    _persist_global_cache()
    dprint("Copying cache back to S3...")
    subprocess.run(
        [
            "rsync",
            "-ahSD",
            "--no-whole-file",
            "--no-compress",
            "--stats",
            "--inplace",
            f"{os.environ['CACHE_DIR']}/",
            env.channel_input_dirs["cache"],
        ]
    )


def standalone_params(env: environment.Environment):
    from cloudpathlib import CloudPath

    train_data = Path(env.channel_input_dirs["train"])
    prior_data = Path(env.channel_input_dirs["prior"])
    output_data = Path(env.channel_input_dirs["output"])

    id = os.environ["DREAMBOOTH_ID"]
    bucket = CloudPath(os.environ["DREAMBOOTH_BUCKET"])

    (bucket / "dataset" / id).download_to(train_data)
    if not prior_data.exists():
        hash = hash_bytes(HyperParams().prior_prompt.encode())
        (bucket / "priors" / hash).download_to(prior_data)

    params = get_params()
    params.prior_class = Class(prompt_=params.prior_prompt, data=prior_data)
    params.image_output_path = output_data
    params.model_output_path = Path(env.model_dir)

    return {"instance_path": train_data, "params": params}


def standalone_cleanup(env: environment.Environment):
    from cloudpathlib import CloudPath

    id = os.environ["DREAMBOOTH_ID"]
    bucket = CloudPath(os.environ["DREAMBOOTH_BUCKET"])

    (bucket / "output" / id).upload_from(env.channel_input_dirs["output"])


def main():
    env = environment.Environment()
    env.is_main = os.getenv("LOCAL_RANK", "-1") == "0"

    if env.channel_input_dirs:
        params = sagemaker_params(env)
        cleanup_fn = sagemaker_cleanup
    elif id is not None:
        params = standalone_params(env)
        cleanup_fn = standalone_cleanup
    else:
        raise ValueError("No input data provided!")

    model = get_model(**params)
    model.accelerator.wait_for_everyone()

    try:
        pipeline = model.train()
        model.eval(pipeline)
    finally:
        if env.is_main:
            model.accelerator.end_training()
            cleanup_fn(env)

        dprint("Exiting!")

    sys.exit(0)


if __name__ == "__main__":
    main()
