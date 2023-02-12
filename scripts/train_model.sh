#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

accelerate launch train/train.py
