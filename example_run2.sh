#!/bin/bash

# Create another script that will save the model outputs for at least one image
# - Eval file needs to make a new model object
# - the state_dict has all of the weights
# - Use that info to get the model
#     - Model created
#     - Load weights
#     - Run on one file
#     - Batch and channel (channel is 1)

# run the runner.py script here
# save the values of state_dict from runner_2.py
# use those weights to make a model
# run that model on one file/image

# cd /data/vision/polina/scratch/haleysan/sha-gen

source /data/vision/polina/scratch/dmoyer/for_haley.sh

config="configs/exp_${SLURM_ARRAY_TASK_ID}.toml"

# need to run runner_2.py on server? if so how within a bash script?
# PYTHONPATH="src/:/data/vision/polina/scratch/fetal-reorg/src/:${PYTHONPATH}" python \
#   src/runner_2.py \
#   --config ${config} #configs/test-v3.toml # > output.txt

PYTHONPATH="src/:/data/vision/polina/scratch/fetal-reorg/src/:${PYTHONPATH}" python \
  eval.py \
  --config ${config} 
# use the output in the output.txt that has the state_dict to use to load into the model
