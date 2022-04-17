#!/bin/bash
#SBATCH --job-name=roberta_hyperparameter_search
#SBATCH --open-mode=append
#SBATCH --output=/scratch/aml731/%j_%x.out
#SBATCH --error=/scratch/aml731/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=12G
#SBATCH -c 4

## TODO: Run your hyperparameter search in a singularity container. Remember to 
## activate your Conda environment in the container before running your Python script.

singularity exec --nv --bind /scratch --overlay /scratch/aml731/overlay-25GB-500K.ext3:rw /scratch/work/public/singularity/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c \
"
source /ext3/env.sh

conda activate 

python /scratch/$USER/AggregateLMs/RoBERTa/run_hyperparameter_search.py -d /scratch/$USER/superglue/RTE
"
