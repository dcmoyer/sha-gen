#!/bin/bash

#SOURCE THE APPROPRIATE PYTHON PATH HERE
#source /data/vision/polina/scratch/dmoyer/bash_source.sh

PYTHONPATH="src/:${PYTHONPATH}" python \
  scripts/runner.py \
  --config configs/test-v3.toml





