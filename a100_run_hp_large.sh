#!/bin/bash
mkdir -p /scratch/$USER/outputs
mkdir -p /scratch/$USER/outputs/models
mkdir -p /scratch/$USER/outputs/model_checkpoints
mkdir -p /scratch/$USER/outputs/model_checkpoints/$3
mkdir -p /scratch/$USER/outputs/model_checkpoints/$3/$2
mkdir -p /scratch/$USER/AggregateLMs/slurm_logs
mkdir -p /scratch/$USER/AggregateLMs/slurm_logs/$3
mkdir -p /scratch/$USER/AggregateLMs/slurm_logs/$3/$2
echo "Task: $2. Requested $5:00:00 hours, $6G of mem, with singularity overlay /scratch/$USER/overlay-25GB-500K$4.ext3 on GPU $7-$8. Not using $4 argument"

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=tune_"$2"
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/"$3"/"$2"/%j_%x.out
#SBATCH --error=slurm_logs/"$3"/"$2"/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=$5:00:00
#SBATCH --gres=gpu:$7:$8
#SBATCH --mem=$6G
#SBATCH -c 4

singularity exec --nv --overlay /scratch/$USER/overlay-25GB-500Ka4.ext3:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "
source /ext3/env.sh
conda activate /scratch/pfi203/envs_dirs/a100gpu2
python /scratch/$USER/AggregateLMs/$1/a100_runhp_large.py -d /scratch/$USER/superglue/$2 -o /scratch/$USER/outputs/model_checkpoints -m $3

"
EOT