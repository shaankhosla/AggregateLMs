#!/bin/bash
#SBATCH --job-name=variance
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
python /scratch/ms12768/AggregateLMs/run_variance_analysis.py 