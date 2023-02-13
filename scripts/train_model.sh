#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

accelerate launch dreambooth/train/train.py
