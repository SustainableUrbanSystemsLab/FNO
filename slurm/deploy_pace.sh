#!/bin/bash

# Georgia Tech PACE Deployment Wrapper
# Usage: bash slurm/deploy_pace.sh --gpu [H200|RTX6000]

GPU_TYPE="h200"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu) GPU_TYPE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Convert to lowercase for matching
GPU_TYPE_LOWER=$(echo "$GPU_TYPE" | tr '[:upper:]' '[:lower:]')

case $GPU_TYPE_LOWER in
    h200)
        SLURM_GPU="h200:1"
        ;;
    rtx6000)
        SLURM_GPU="rtx6000:1"
        ;;
    *)
        echo "Error: Unsupported GPU type '$GPU_TYPE'."
        echo "Supported types: H200, RTX6000"
        exit 1
        ;;
esac

echo "=========================================="
echo " Preparing PACE Deployment"
echo " GPU Requested: $GPU_TYPE ($SLURM_GPU)"
echo "=========================================="

# Create logs directory locally to avoid sbatch errors
mkdir -p logs

# Submit the job
# We pass the gres directly here. 
# You might need to add --account=YOUR_ACCOUNT or --partition=YOUR_PARTITION 
# if PACE requires them for your specific allocation.
sbatch --gres=gpu:$SLURM_GPU slurm/pace_train.sbatch

echo "------------------------------------------"
echo "Job submitted. Check status with: squeue -u $USER"
echo "Logs will be in the 'logs/' directory."
