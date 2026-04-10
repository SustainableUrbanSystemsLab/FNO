#!/bin/bash
#SBATCH -J fno_hybrid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH -t 11:50:00
#SBATCH -o logs/hybrid_%j.out
#SBATCH -e logs/hybrid_%j.err
#SBATCH --gres=gpu:H200:1
#SBATCH --account=coa
#SBATCH --partition=ice-gpu

echo "Job started on $(date)"
echo "Running on node $(hostname)"

cd $SLURM_SUBMIT_DIR
mkdir -p logs

# Remove Windows .venv if uploaded from local machine
if [ -d ".venv/Scripts" ]; then
    echo "Detected Windows .venv. Removing it..."
    rm -rf .venv
fi

# Install uv if not available
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Syncing dependencies..."
uv sync --frozen --no-cache

# Optional flag pass-through (set via sbatch --export or deploy_ice.sh)
CONFIG_FLAG="--config ${CONFIG_FILE:-config.toml}"
FRESH_FLAG="${FRESH_TRAIN:+--fresh}"
RESET_FLAG="${RESET_PATIENCE:+--reset-patience}"

echo "Starting HybridFNO training..."
uv run python pipelines/train/hybrid.py $CONFIG_FLAG $FRESH_FLAG $RESET_FLAG

echo "Job finished on $(date)"
