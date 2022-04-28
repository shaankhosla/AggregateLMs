#!/bin/bash
#SBATCH --job-name=nlu_model_bagging
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=20G
#SBATCH -c 4

singularity exec --nv --overlay /scratch/$USER/overlay-25GB-500K_2.ext3:rw /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "
source /ext3/env.sh
conda activate
python /scratch/$USER/AggregateLMs/model_roberta/train_models.py -d /scratch/$USER/superglue/BoolQ -o /scratch/$USER/outputs -s /scratch/$USER/outputs/bootstrapped_models -e 5 -t 32 -lr 0.0002498160 -w 0.1903892 -n 10

"