#!/bin/bash

## honestly not sure if you call a shell script to call a python file where the CWD is... it looks like its where the shell script is but gotta check

mkdir -p /scratch/$USER/outputs
mkdir -p /scratch/$USER/outputs/models
mkdir -p /scratch/$USER/outputs/model_checkpoints

pip install -r /scratch/$USER/AggregateLMs/requirements.txt
bash /scratch/$USER/AggregateLMs/download_data.sh
python $1/run_hyperparameter_search.py -d /scratch/$USER/superglue/$2 -o /scratch/$USER/outputs/model_checkpoints
