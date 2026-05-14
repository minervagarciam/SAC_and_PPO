#!/bin/bash
#SBATCH --account=aip-ashique
#SBATCH -o logs/job-%A-%a.out
#SBATCH -e logs/job-%A-%a.err
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-19

export MUJOCO_GL=disabled

# Load modules
module load python/3.11
module load mujoco

# Activate virtual environment
source venv/bin/activate

# Run ppo
parallel --jobs 2 -u :::\
    "python ppo_continuous_action.py \
        --env_id dm_control/hopper-stand-v0 \
        --seed $SLURM_ARRAY_TASK_ID \
        --total_timesteps 10000000" \
    "python ppo_continuous_action.py \
        --env_id dm_control/cheetah-run-v0 \
        --seed $SLURM_ARRAY_TASK_ID \
        --total_timesteps 10000000" 