#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=tune_"$2"
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/"$3"/"$2"/%j_%x.out
#SBATCH --error=slurm_logs/"$3"/"$2"/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:3
#SBATCH --mem=40G
#SBATCH -c 4

singularity exec --nv --overlay /scratch/$USER/overlay-25GB-500K.ext3:rw /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "
source /ext3/env.sh
conda activate
bash run_model_task.sh /scratch/$USER/AggregateLMs/$1 $2 $3

"
EOT