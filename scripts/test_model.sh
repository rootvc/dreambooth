#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

accelerate launch train/tests/test_model.py
