#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

accelerate launch tests/test_model.py
