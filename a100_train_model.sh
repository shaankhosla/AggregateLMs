#!/bin/bash
mkdir -p /scratch/$USER/saved_models
mkdir -p /scratch/$USER/saved_models/models
mkdir -p /scratch/$USER/saved_models/model_checkpoints
mkdir -p /scratch/$USER/saved_models/logs
mkdir -p /scratch/$USER/saved_models/logs/$3
mkdir -p /scratch/$USER/saved_models/logs/$3/$2
echo "Task $2 with model $3. Requested $5:00:00 hours, $6G of mem with gpu $7-$8. No longer using $4 argument"

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name="$3"_"$2"
#SBATCH --open-mode=append
#SBATCH --output=/scratch/${USER}/saved_models/logs/$3/$2/%j_%x.out
#SBATCH --error=/scratch/${USER}/saved_models/logs/$3/$2/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=$5:00:00
#SBATCH --gres=gpu:$7:$8
#SBATCH --mem=$6G
#SBATCH -c 4

singularity exec --nv --overlay /scratch/$USER/overlay-25GB-500Ka4.ext3:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "
source /ext3/env.sh
conda activate /scratch/pfi203/envs_dirs/a100gpu2
python /scratch/${USER}/AggregateLMs/$1/train_models.py -d /scratch/${USER}/superglue/$2  -o /scratch/${USER}/saved_models/model_checkpoints -s /scratch/${USER}/saved_models/models -m $3 -n 10

"
EOT