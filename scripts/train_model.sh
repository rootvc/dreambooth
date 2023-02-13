#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

CONFIG="dreambooth/config/accelerate/${INSTANCE_TYPE}.yml"
if [ -f "${CONFIG}"]; then
  cp -f "${CONFIG}" ~/.cache/huggingface/accelerate/default_config.yaml
fi

accelerate launch dreambooth/train/train.py
