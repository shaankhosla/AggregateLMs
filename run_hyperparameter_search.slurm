#!/bin/bash
#SBATCH --job-name=nlu_hyperparameter_tuning
#SBATCH --open-mode=append
#SBATCH --output=/scratch/aml731/nlu_%j_%x.out
#SBATCH --error=/scratch/aml731/nlu_%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH -c 1

singularity exec --nv --overlay /scratch/$USER/overlay-25GB-500K.ext3:rw /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "
source /ext3/env.sh
conda activate
bash run_model_task.sh /scratch/$USER/AggregateLMs/model_roberta CB

"