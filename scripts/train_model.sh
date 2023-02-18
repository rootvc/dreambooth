#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

echo "Training model from $PWD"
ls -la
ls -la opt/ml/input/data/model

CONFIG_ROOT="config/accelerate"
DEFAULT_CONFIG="${CONFIG_ROOT}/default.yml"
CONFIG="${CONFIG_ROOT}/${INSTANCE_TYPE}.yml"

if [ -f "${CONFIG}" ]; then
  cp -f "${CONFIG}" ~/.cache/huggingface/accelerate/default_config.yaml
else
  cp -f "${DEFAULT_CONFIG}" ~/.cache/huggingface/accelerate/default_config.yaml
fi

accelerate launch dreambooth/train/train.py
