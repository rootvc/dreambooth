import logging
import sys

import diffusers
import tensorflow as tf
import torch
import torch._dynamo.config
import torch.backends.cuda
import torch.backends.cudnn
import transformers
from loguru._logger import Logger
from modal import is_local


def init_logging(logger: Logger):
    logger.remove()
    logger.add(
        sys.stderr,
        level=logging.INFO,
        format="+{elapsed} {name}[{level.icon}]: {message}",
        backtrace=True,
        diagnose=True,
        colorize=True,
    )


def init_config(split_gpus=True):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch._dynamo.config.suppress_errors = True

    tf.get_logger().setLevel(logging.INFO)

    if split_gpus and not is_local():
        gpus = tf.config.list_physical_devices("GPU")
        assert len(gpus) > 1
        tf.config.set_visible_devices(gpus[-1], "GPU")
        devices = tf.config.list_logical_devices("GPU")
        assert len(devices) == len(gpus) - 1

    transformers.utils.logging.set_verbosity_info()
    diffusers.utils.logging.set_verbosity_info()
