import logging
import sys

import diffusers
import torch
import torch._dynamo.config
import torch.backends.cuda
import torch.backends.cudnn
import transformers
from diffusers.utils.logging import disable_progress_bar
from loguru import logger as default_logger
from loguru._logger import Logger
from modal import is_local


def init_logging(logger: Logger = default_logger):
    logger.remove()

    def format(record):
        mins = record["elapsed"].seconds // 60
        secs = record["elapsed"].seconds % 60
        rank = record["extra"].get("rank", 0)
        return (
            f"+{mins}m{secs}s"
            + " %s[{level.icon}] {message} ({name})\n{exception}" % rank
        )

    logger.add(
        sys.stderr,
        level=logging.INFO,
        format=format,
        backtrace=True,
        diagnose=True,
        colorize=True,
        enqueue=True,
    )


def init_torch_config():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.cache_size_limit = 128

    disable_progress_bar()
    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_warning()


def init_config(split_gpus=True):
    init_torch_config()

    import tensorflow as tf

    tf.get_logger().setLevel(logging.DEBUG)
    tf.config.optimizer.set_jit("autoclustering")

    if split_gpus and not is_local():
        gpus = tf.config.list_physical_devices("GPU")
        assert len(gpus) > 1
        tf.config.set_visible_devices(gpus[-1], "GPU")
        devices = tf.config.list_logical_devices("GPU")
        assert len(devices) == 1
