#!/bin/bash
# Run this script ONCE on Compute Canada to set up your Python environment
# Usage: bash setup_env.sh

# Navigate to your project folder
cd /home/garciam1/projects/aip-ashique/garciam1/SAC_and_PPO

# Load the required CC modules
module load python/3.11

# Create a virtual environment
virtualenv --no-download venv

# Activate it
source venv/bin/activate

# Upgrade pip
pip install --no-index --upgrade pip

# Install required packages
pip install torch --no-index
pip install gymnasium
pip install shimmy[dm-control]
pip install tyro
pip install numpy

echo "Setup complete! Virtual environment is ready."
