#!/bin/bash

# ICE Deployment Wrapper
# Usage: bash slurm/deploy_ice.sh --gpu [H100|A100] [--ngpus 1|2] [--config config.toml]

GPU_TYPE="h200" # ICE often has H100s or A100s

# Defaults
NUM_GPUS=2
CONFIG_FILE="config.toml"
RESET_PATIENCE=""
FRESH_TRAIN=""
TRAIN_SCRIPT="pipelines/train/distributed.py"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu) GPU_TYPE="$2"; shift ;;
        --ngpus) NUM_GPUS="$2"; shift ;;
        --config) CONFIG_FILE="$2"; shift ;;
        --script) TRAIN_SCRIPT="$2"; shift ;;
        --reset-patience) RESET_PATIENCE="1" ;;
        --fresh) FRESH_TRAIN="1" ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Convert to lowercase
GPU_TYPE_LOWER=$(echo "$GPU_TYPE" | tr '[:upper:]' '[:lower:]')

# Map to correct ICE gres names if known, otherwise pass through
# ICE usually uses standard gres like gpu:h100:1 or gpu:a100:1
case $GPU_TYPE_LOWER in
    h200) SLURM_GPU="H200:${NUM_GPUS}" ;;
    h100) SLURM_GPU="H100:${NUM_GPUS}" ;;
    a100) SLURM_GPU="A100:${NUM_GPUS}" ;;
    v100) SLURM_GPU="V100:${NUM_GPUS}" ;;
    *) SLURM_GPU="${GPU_TYPE}:${NUM_GPUS}" ;;
esac

# Read ICE config from config.toml
if [ -f "$CONFIG_FILE" ]; then
    # Look for [ice] section first
    ICE_ACCOUNT=$(sed -n '/^\[ice\]/,/^\[/p' "$CONFIG_FILE" | grep -E "^account\s*=" | sed 's/.*=\s*"\(.*\)".*/\1/' | tr -d ' ')
    ICE_PARTITION=$(sed -n '/^\[ice\]/,/^\[/p' "$CONFIG_FILE" | grep -E "^partition\s*=" | sed 's/.*=\s*"\(.*\)".*/\1/' | tr -d ' ')
else
    echo "Warning: $CONFIG_FILE not found."
    ICE_ACCOUNT="coc" # Default fallback
    ICE_PARTITION=""
fi

# Fallback if parsing failed
if [ -z "$ICE_ACCOUNT" ]; then
    ICE_ACCOUNT="coc"
fi

echo "=========================================="
echo " Preparing ICE Deployment"
echo " GPU Requested: $GPU_TYPE x $NUM_GPUS ($SLURM_GPU)"
echo " Config file: $CONFIG_FILE"
echo " Account: $ICE_ACCOUNT"
echo "=========================================="

mkdir -p logs

# Build sbatch command
# Note: ICE might need specific partition if gpu is not default
SBATCH_CMD="sbatch --gres=gpu:$SLURM_GPU --account=$ICE_ACCOUNT --export=NUM_GPUS=$NUM_GPUS,CONFIG_FILE=$CONFIG_FILE,RESET_PATIENCE=$RESET_PATIENCE,FRESH_TRAIN=$FRESH_TRAIN,TRAIN_SCRIPT=$TRAIN_SCRIPT"

if [ -n "$ICE_PARTITION" ]; then
    SBATCH_CMD="$SBATCH_CMD --partition=$ICE_PARTITION"
fi

# Reuse the same sbatch script as it is generic enough (just needs correct account/gres)
# IMPORTANT: Check if pace_train.sbatch works for ICE or if we need a copy.
# pace_train.sbatch loads 'uv' and runs 'torchrun'. This should be fine on ICE too.
SBATCH_CMD="$SBATCH_CMD slurm/pace_train.sbatch"

# Submit
$SBATCH_CMD

echo "------------------------------------------"
echo "Job submitted on ICE. Check status with: squeue -u $USER"
