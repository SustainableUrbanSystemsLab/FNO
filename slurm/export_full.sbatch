#!/bin/bash
#SBATCH -J export_data
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o logs/export_data_%j.out
#SBATCH -e logs/export_data_%j.err

echo "Job started on $(date)"
cd $SLURM_SUBMIT_DIR
mkdir -p logs

# Sync dependencies
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv sync --frozen --no-cache

echo "Starting full dataset export..."
uv run python export_full_dataset.py

echo "Job finished on $(date)"
