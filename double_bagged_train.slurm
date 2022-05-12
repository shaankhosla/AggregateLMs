#!/bin/bash
#SBATCH --job-name=nlu_hyperparameter_tuning
#SBATCH --open-mode=append
#SBATCH --output=new_slurm_logs/%j_%x.out
#SBATCH --error=new_slurm_logs/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=07:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=20G
#SBATCH -c 4
#SBATCH --mail-type=END
#SBATCH --mail-user=ms12768@nyu.edu

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate bash
python /scratch/ms12768/AggregateLMs/model_deberta/train_models_variance.py -d /scratch/ms12768/superglue/BoolQ -o /scratch/ms12768/outputs/saved_models/model_checkpoints -s /scratch/ms12768/double_bootstrapped_models/deberta-large/BoolQ -m microsoft/deberta-v3-large  -n 10