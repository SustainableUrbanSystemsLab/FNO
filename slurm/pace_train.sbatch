#!/bin/bash
#SBATCH -J fno_train
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -o logs/fno_%j.out
#SBATCH -e logs/fno_%j.err

# Note: --gres=gpu:xxx:1 is passed from the deploy_pace.sh script

echo "Job started on $(date)"
echo "Running on node $(hostname)"

# Load python module or use system python if uv handles it
module load anaconda3/2023.09 2>/dev/null || echo "Module load failed, continuing with system python..."

# Go to project root (one level up from slurm/)
cd ..

# Ensure logs directory exists
mkdir -p logs

# Setup/Ensure environment
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Running setup..."
    bash slurm/pace_setup.sh
fi

source .venv/bin/activate

# Execute training
echo "Starting training script..."
python train_fno_mag.py

echo "Job finished on $(date)"
