#!/bin/bash

#SOURCE THE APPROPRIATE PYTHON PATH HERE
#source /data/vision/polina/scratch/dmoyer/bash_source.sh

# PYTHONPATH="src/:/data/vision/polina/scratch/fetal-reorg/src/:${PYTHONPATH}" python \
#   src/runner_2.py \
#   --config configs/test-v3.toml


PYTHONPATH="src/:/data/vision/polina/scratch/fetal-reorg/src/:${PYTHONPATH}" python \
  src/eval.py 


