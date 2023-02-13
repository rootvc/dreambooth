#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.
cp /root/.cache/huggingface/accelerate/default_config.yaml ~/.cache/huggingface/accelerate/default_config.yaml

accelerate launch dreambooth/train/train.py
