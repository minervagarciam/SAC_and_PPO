#!/bin/bash
#SBATCH --account=aip-ashique
#SBATCH -o logs/job-%A-%a.out
#SBATCH -e logs/job-%A-%a.err
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1
#SBATCH --array=0-9

export MUJOCO_GL=disabled

# Load modules
module load python/3.11
module load mujoco

# Activate virtual environment
source venv/bin/activate

echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Python: $(which python)"
echo "Working dir: $(pwd)"

# Run SAC
parallel --jobs 5 -u :::\
"python sac_continuous_action.py \
    --env_id 'dm_control/pendulum-swingup-v0' \
    --seed $SLURM_ARRAY_TASK_ID \
    --total_timesteps 1000000"\
"python sac_continuous_action.py \
    --env_id 'dm_control/cartpole-swingup-v0' \
    --seed $SLURM_ARRAY_TASK_ID \
    --total_timesteps 1000000" \
"python sac_continuous_action.py \
    --env_id 'dm_control/reacher-easy-v0' \
    --seed $SLURM_ARRAY_TASK_ID \
    --total_timesteps 1000000" \
"python sac_continuous_action.py \
    --env_id 'dm_control/hopper-stand-v0' \
    --seed $SLURM_ARRAY_TASK_ID \
    --total_timesteps 1000000" \
"python sac_continuous_action.py \
    --env_id 'dm_control/cheetah-run-v0' \
    --seed $SLURM_ARRAY_TASK_ID \
    --total_timesteps 1000000" 