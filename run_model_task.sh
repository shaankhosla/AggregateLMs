#!/bin/bash

pip install -r /scratch/$USER/AggregateLMs/requirements.txt
bash /scratch/$USER/AggregateLMs/download_data.sh
python $1/run_hyperparameter_search.py -d /scratch/$USER/superglue/$2 -o /scratch/$USER/outputs/model_checkpoints -m $3
