#!/bin/bash
#SBATCH --job-name=nlu_hyperparameter_tuning
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH -c 1

singularity exec --nv --overlay /scratch/$USER/overlay-25GB-500K.ext3:rw /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "
source /ext3/env.sh
conda activate
python /scratch/pfi203/AggregateLMs/model_roberta/train_models.py -d /scratch/pfi203/superglue/BoolQ  -o /scratch/pfi203/outputs/model_checkpoints -s /scratch/pfi203/outputs/saved_models -e 3 -t 16 -lr 0.005 -w 0.001 -n 3
"