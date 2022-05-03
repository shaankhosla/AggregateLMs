#!/bin/bash
#SBATCH --job-name=nlu_model_bagging
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH -c 4

singularity exec --nv --overlay /scratch/$USER/overlay-25GB-500K.ext3:rw /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "
source /ext3/env.sh
conda activate
python /scratch/$USER/AggregateLMs/model_distilroberta/train_models.py -d /scratch/$USER/superglue/BoolQ -o /scratch/$USER/outputs -s /scratch/$USER/outputs/bootstrapped_models -e 7 -t 16 -lr 0.00002498160 -w 0.1903892 -n 10

"