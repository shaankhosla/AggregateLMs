#!/bin/bash
#SBATCH --job-name=nlu_model_bagging
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --mem=40G
#SBATCH -c 4

singularity exec --nv --overlay /scratch/$USER/overlay-25GB-500K_3.ext3:rw /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "
source /ext3/env.sh
conda activate
python /scratch/$USER/AggregateLMs/model_roberta/train_models.py -m roberta-large -d /scratch/$USER/superglue/BoolQ -o /scratch/$USER/outputs -s /scratch/$USER/outputs/bootstrapped_models -e 7 -t 16 -lr 0.00001624074 -w 0.0354189 -n 10

"