#!/bin/bash
#SBATCH -J plot_dist
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o logs/plot_dist_%j.out
#SBATCH -e logs/plot_dist_%j.err

echo "Job started on $(date)"
echo "Running on node $(hostname)"

# Go to the directory where sbatch was submitted from
cd $SLURM_SUBMIT_DIR

# Ensure logs directory exists
mkdir -p logs

# Install uv if not available (standard setup)
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Sync dependencies
echo "Syncing dependencies..."
uv sync --frozen --no-cache

echo "Starting distribution plot..."
# Run the plot script
uv run python plot_dataset_distribution.py

echo "Job finished on $(date)"
