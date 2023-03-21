import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from functools import partial
from pathlib import Path
from typing import TypedDict

import torch
import torch.version
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

    print(f"Unpacked {name} to {model_dir}...")
    return model_dir


def _unpack_eval_models(model_dir: Path):
    import facelib.utils.misc

    params = get_params()
    models = model_dir / params.eval_model_path
    facelib_path = Path(facelib.utils.misc.ROOT_DIR) / "weights"

    if not models.exists():
        raise ValueError(f"Model directory {models} does not exist!")

    if not Path("weights").is_symlink():
        os.symlink(models, "weights", target_is_directory=True)

    if not facelib_path.is_symlink():
        shutil.rmtree(facelib_path, ignore_errors=True)
        os.symlink(models, facelib_path, target_is_directory=True)


def _setup_global_cache():
    import torch._inductor.config

    torch._inductor.config.global_cache_path = (
        Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]) / "global_cache"
    )


def _persist_global_cache():
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
        dprint("No global cache found, creating...")
        global_cache_data = {dinfo: {vinfo: {}}}
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
    params.test_model = _unpack_model(env, params.test_model)
    params.prior_class = Class(prompt_=params.prior_prompt, data=prior_data)
    params.image_output_path = output_data
    params.model_output_path = Path(env.model_dir)

    _setup_global_cache()
    if env.is_main:
        shutil.copytree(cache_data, os.environ["CACHE_DIR"], dirs_exist_ok=True)
        _unpack_eval_models(Path(env.channel_input_dirs["model"]))

    return {"instance_path": train_data, "params": params}


def sagemaker_cleanup(env: environment.Environment):
    # No need to persist models
    shutil.rmtree(env.model_dir, ignore_errors=True)
    os.makedirs(env.model_dir, exist_ok=True)

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


def cache_path():
    from cloudpathlib import CloudPath

    bucket = CloudPath(os.environ["DREAMBOOTH_BUCKET"])

    dinfo = torch.cuda.get_device_properties(torch.cuda.current_device()).name
    vinfo = torch.version.cuda
    key = f"{dinfo}-{vinfo}-{torch.__version__}".replace(" ", "-").lower()
    return bucket / "cache" / key


def standalone_params(is_main: bool) -> Params:
    from cloudpathlib import CloudPath

    base_dir = Path("/opt/ml")
    train_data = base_dir / "train"
    prior_data = base_dir / "prior"
    output_data = base_dir / "output"
    model_data = base_dir / "model"

    id = os.environ["DREAMBOOTH_ID"]
    bucket = CloudPath(os.environ["DREAMBOOTH_BUCKET"])

    params = get_params()
    params.prior_class = Class(prompt_=params.prior_prompt, data=prior_data)
    params.image_output_path = output_data
    params.model_output_path = Path(model_data)

    params.model_output_path.mkdir(parents=True, exist_ok=True)
    params.image_output_path.mkdir(parents=True, exist_ok=True)

    if os.getenv("WARM", "0") == "1":
        params.ti_train_epochs = 1
        params.train_epochs = 2
        params.validate_after_steps = 0
        params.validate_every_epochs = None
        params.eval_prompts = params.eval_prompts[:1]

        dprint("Downloading cache from S3...")
        subprocess.run(
            [
                "rclone",
                "sync",
                "--checksum",
                "--fast-list",
                "--human-readable",
                "--stats-one-line",
                "--transfers=8",
                "--s3-env-auth",
                "--s3-region=us-west-2",
                "--s3-use-accelerate-endpoint",
                "--s3-no-check-bucket",
                "--s3-no-head",
                f"{cache_path()}/",
                f"{os.environ['CACHE_DIR']}",
            ]
        )

    if not is_main:
        return {"instance_path": train_data, "params": params}

    train_data_path = bucket / "dataset" / id
    priors_path = bucket / "priors" / hash_bytes(HyperParams().prior_prompt.encode())

    for src, dst in {train_data_path: train_data, priors_path: prior_data}.items():
        if src.exists() and not dst.exists():
            src.download_to(dst)

    if hf_model_cache := os.getenv("HF_MODEL_CACHE"):
        cache = Path(hf_model_cache)
        params.model.name = cache / params.model.name
        if params.model.vae:
            params.model.vae = cache / params.model.vae
        params.test_model = cache / params.test_model

        _unpack_eval_models(Path(hf_model_cache))

    _setup_global_cache()

    return {"instance_path": train_data, "params": params}


def standalone_cleanup():
    from cloudpathlib import CloudPath

    id = os.environ["DREAMBOOTH_ID"]
    bucket = CloudPath(os.environ["DREAMBOOTH_BUCKET"])

    (bucket / "output" / id).upload_from(
        Path("/opt/ml/output"), force_overwrite_to_cloud=True
    )

    dprint("Persisting global cache...")
    _persist_global_cache()
    dprint("Copying cache back to S3...")
    subprocess.run(
        [
            "rclone",
            "sync",
            "--checksum",
            "--fast-list",
            "--human-readable",
            "--stats-one-line",
            "--transfers=8",
            "--s3-upload-concurrency=8",
            "--s3-env-auth",
            "--s3-region=us-west-2",
            "--s3-use-accelerate-endpoint",
            "--s3-no-check-bucket",
            "--s3-no-head",
            f"{os.environ['CACHE_DIR']}/",
            cache_path(),
        ]
    )


def main():
    is_main = os.getenv("LOCAL_RANK", "0") == "0"
    is_sagemaker = "SAGEMAKER_JOB_NAME" in os.environ

    dprint("Starting!", reset=True)

    if is_sagemaker:
        env = environment.Environment()
        env.is_main = is_main
        params = sagemaker_params(env)
        cleanup_fn = partial(sagemaker_cleanup, env)
    elif os.getenv("DREAMBOOTH_ID") is not None:
        params = standalone_params(is_main)
        cleanup_fn = standalone_cleanup
    else:
        raise ValueError("No input data provided!")

    model = get_model(**params)
    model.accelerator.wait_for_everyone()

    try:
        pipeline = model.train()
        model.eval(pipeline)
    finally:
        if is_main:
            model.accelerator.end_training()
            cleanup_fn()

        dprint("Exiting!")

    if is_sagemaker:
        sys.exit(0)


if __name__ == "__main__":
    main()
