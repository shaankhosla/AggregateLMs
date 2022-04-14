#!/bin/bash

## honestly not sure if you call a shell script to call a python file where the CWD is... it looks like its where the shell script is but gotta check
mkdir outputs
mkdir $1/outputs

pip install -r $1/requirements.txt
bash $1/download_data.sh
python $1/run_hyperparameter_search.py -d /tmp/superglue/$2