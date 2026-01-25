#!/bin/bash

# Georgia Tech PACE Deployment Wrapper
# Usage: bash slurm/deploy_pace.sh --gpu [H200|RTX6000] [--ngpus 1|2]

GPU_TYPE="h200"
NUM_GPUS=2  # Default to 2 GPUs for distributed training

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu) GPU_TYPE="$2"; shift ;;
        --ngpus) NUM_GPUS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Convert to lowercase for matching
GPU_TYPE_LOWER=$(echo "$GPU_TYPE" | tr '[:upper:]' '[:lower:]')

case $GPU_TYPE_LOWER in
    h200)
        SLURM_GPU="h200:${NUM_GPUS}"
        ;;
    rtx6000)
        SLURM_GPU="rtx6000:${NUM_GPUS}"
        ;;
    *)
        echo "Error: Unsupported GPU type '$GPU_TYPE'."
        echo "Supported types: H200, RTX6000"
        exit 1
        ;;
esac

# Read PACE config from config.toml
CONFIG_FILE="config.toml"
if [ -f "$CONFIG_FILE" ]; then
    PACE_ACCOUNT=$(grep -E "^account\s*=" "$CONFIG_FILE" | sed 's/.*=\s*"\(.*\)".*/\1/' | tr -d ' ')
    PACE_PARTITION=$(grep -E "^partition\s*=" "$CONFIG_FILE" | sed 's/.*=\s*"\(.*\)".*/\1/' | tr -d ' ')
else
    echo "Warning: $CONFIG_FILE not found. Set account in config.toml."
    PACE_ACCOUNT=""
    PACE_PARTITION=""
fi

# Validate account
if [ -z "$PACE_ACCOUNT" ]; then
    echo "Error: PACE account not set in config.toml"
    echo "Edit config.toml and set: account = \"gts-yourusername\""
    exit 1
fi

echo "=========================================="
echo " Preparing PACE Deployment"
echo " GPU Requested: $GPU_TYPE x $NUM_GPUS ($SLURM_GPU)"
echo " Account: $PACE_ACCOUNT"
echo "=========================================="

# Create logs directory locally to avoid sbatch errors
mkdir -p logs

# Build sbatch command
SBATCH_CMD="sbatch --gres=gpu:$SLURM_GPU --account=$PACE_ACCOUNT --export=NUM_GPUS=$NUM_GPUS"
if [ -n "$PACE_PARTITION" ]; then
    SBATCH_CMD="$SBATCH_CMD --partition=$PACE_PARTITION"
fi
SBATCH_CMD="$SBATCH_CMD slurm/pace_train.sbatch"

# Submit the job
$SBATCH_CMD

echo "------------------------------------------"
echo "Job submitted. Check status with: squeue -u $USER"
echo "Logs will be in the 'logs/' directory."
