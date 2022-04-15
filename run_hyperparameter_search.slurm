#!/bin/bash
#SBATCH --job-name=testproj2
#SBATCH --open-mode=append
#SBATCH --output=/scratch/pfi203/outputs/%j_%x.out
#SBATCH --error=/scratch/pfi203/outputs/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=28G
#SBATCH -c 4

singularity exec --nv --overlay /scratch/pfi203/overlay-25GB-500K.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "
source /ext3/env.sh
conda activate
bash run_model_task.sh model_gpt2 CB
"