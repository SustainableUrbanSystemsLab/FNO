#!/bin/bash
#SBATCH -J fno_train
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH -t 4:00:00
#SBATCH -o logs/fno_%j.out
#SBATCH -e logs/fno_%j.err

# Note: --gres=gpu:xxx:N is passed from the deploy_pace.sh script
# NUM_GPUS is also passed via --export

echo "Job started on $(date)"
echo "Running on node $(hostname)"
echo "Number of GPUs: ${NUM_GPUS:-1}"

# Go to the directory where sbatch was submitted from
cd $SLURM_SUBMIT_DIR

# Ensure logs directory exists
mkdir -p logs

# Install uv if not available
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Sync dependencies
echo "Syncing dependencies with uv..."
uv sync

# Execute training based on GPU count
if [ "${NUM_GPUS:-1}" -gt 1 ]; then
    # Use random port to avoid conflicts on shared nodes
    MASTER_PORT=$((29500 + RANDOM % 1000))
    
    # NCCL settings for HPC
    export NCCL_ASYNC_ERROR_HANDLING=1  # Better error handling
    export NCCL_IB_DISABLE=0            # Enable InfiniBand if available
    
    echo "Starting DISTRIBUTED training with ${NUM_GPUS} GPUs on port ${MASTER_PORT}..."
    uv run torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} train_fno_distributed.py
else
    echo "Starting SINGLE-GPU training..."
    uv run python train_fno_distributed.py
fi

echo "Job finished on $(date)"
