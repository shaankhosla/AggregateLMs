#!/bin/bash
mkdir -p /scratch/$USER/outputs
mkdir -p /scratch/$USER/outputs/models
mkdir -p /scratch/$USER/outputs/model_checkpoints
mkdir -p /scratch/$USER/outputs/model_checkpoints/$3
mkdir -p /scratch/$USER/outputs/model_checkpoints/$3/$2
mkdir -p /scratch/$USER/AggregateLMs/slurm_logs
mkdir -p /scratch/$USER/AggregateLMs/slurm_logs/$3
mkdir -p /scratch/$USER/AggregateLMs/slurm_logs/$3/$2
echo "requested $5:00:00 hours, $6G of mem, with singularity overlay /scratch/$USER/overlay-25GB-500K$4.ext3"

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

singularity exec --nv --overlay /scratch/$USER/overlay-25GB-500K$4.ext3:rw /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "
source /ext3/env.sh
conda activate
bash run_model_task_large.sh /scratch/$USER/AggregateLMs/$1 $2 $3

"
EOT