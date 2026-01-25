#!/bin/bash
#SBATCH -J fno_train
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o logs/fno_%j.out
#SBATCH -e logs/fno_%j.err

# Note: --gres=gpu:xxx:N is passed from the deploy_pace.sh script
# NUM_GPUS is also passed via --export

echo "Job started on $(date)"
echo "Running on node $(hostname)"
echo "Number of GPUs: ${NUM_GPUS:-1}"

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

# Execute training based on GPU count
if [ "${NUM_GPUS:-1}" -gt 1 ]; then
    echo "Starting DISTRIBUTED training with ${NUM_GPUS} GPUs..."
    torchrun --nproc_per_node=${NUM_GPUS} train_fno_distributed.py
else
    echo "Starting SINGLE-GPU training..."
    python train_fno_mag.py
fi

echo "Job finished on $(date)"
