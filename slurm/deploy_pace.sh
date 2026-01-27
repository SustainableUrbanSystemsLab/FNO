#!/bin/bash

# Georgia Tech PACE Deployment Wrapper
# Usage: bash slurm/deploy_pace.sh --gpu [H200|RTX6000] [--ngpus 1|2] [--config config_medium.toml]

GPU_TYPE="h200"
NUM_GPUS=2  # Default to 2 GPUs for distributed training
CONFIG_FILE="config.toml"  # Default config (medium model with width=96)
RESET_PATIENCE=""
FRESH_TRAIN=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu) GPU_TYPE="$2"; shift ;;
        --ngpus) NUM_GPUS="$2"; shift ;;
        --config) CONFIG_FILE="$2"; shift ;;
        --config) CONFIG_FILE="$2"; shift ;;
        --reset-patience) RESET_PATIENCE="1" ;;
        --fresh) FRESH_TRAIN="1" ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Convert to lowercase for matching
GPU_TYPE_LOWER=$(echo "$GPU_TYPE" | tr '[:upper:]' '[:lower:]')

# Map to correct PACE gres names (case-sensitive!)
case $GPU_TYPE_LOWER in
    h200)
        SLURM_GPU="H200:${NUM_GPUS}"
        ;;
    rtx6000)
        SLURM_GPU="RTX6000:${NUM_GPUS}"
        ;;
    a100)
        SLURM_GPU="A100:${NUM_GPUS}"
        ;;
    v100)
        SLURM_GPU="V100:${NUM_GPUS}"
        ;;
    *)
        echo "Error: Unsupported GPU type '$GPU_TYPE'."
        echo "Supported types: H200, RTX6000, A100, V100"
        exit 1
        ;;
esac

# Read PACE config from the specified config file
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
    echo "Error: PACE account not set in $CONFIG_FILE"
    echo "Edit $CONFIG_FILE and set: account = \"gts-yourusername\""
    exit 1
fi

echo "=========================================="
echo " Preparing PACE Deployment"
echo " GPU Requested: $GPU_TYPE x $NUM_GPUS ($SLURM_GPU)"
echo " Config file: $CONFIG_FILE"
echo " Account: $PACE_ACCOUNT"
echo "=========================================="

# Create logs directory locally to avoid sbatch errors
mkdir -p logs

# Build sbatch command
SBATCH_CMD="sbatch --gres=gpu:$SLURM_GPU --account=$PACE_ACCOUNT --export=NUM_GPUS=$NUM_GPUS,CONFIG_FILE=$CONFIG_FILE,RESET_PATIENCE=$RESET_PATIENCE,FRESH_TRAIN=$FRESH_TRAIN"
if [ -n "$PACE_PARTITION" ]; then
    SBATCH_CMD="$SBATCH_CMD --partition=$PACE_PARTITION"
fi
SBATCH_CMD="$SBATCH_CMD slurm/pace_train.sbatch"

# Submit the job
$SBATCH_CMD

echo "------------------------------------------"
echo "Job submitted. Check status with: squeue -u $USER"
echo "Logs will be in the 'logs/' directory."
