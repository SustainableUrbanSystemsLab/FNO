#!/bin/bash
# Deploy HybridFNO training on ICE
# Usage: bash slurm/deploy_hybrid.sh [--gpu H200|H100|A100] [--ngpus 1] [--config config.toml] [--fresh] [--reset-patience]
# All cluster credentials (account, partition, walltime) are read from config.toml [ice]

GPU_TYPE="H200"
NUM_GPUS=1
CONFIG_FILE="config.toml"
FRESH_TRAIN=""
RESET_PATIENCE=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu) GPU_TYPE="$2"; shift ;;
        --ngpus) NUM_GPUS="$2"; shift ;;
        --config) CONFIG_FILE="$2"; shift ;;
        --fresh) FRESH_TRAIN="1" ;;
        --reset-patience) RESET_PATIENCE="1" ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Use config.local.toml for [ice] credentials if it exists (gitignored, never overwritten by pulls)
# Falls back to config.toml if not found
LOCAL_CONFIG="config.local.toml"
ICE_CONFIG_FILE="$CONFIG_FILE"
if [ -f "$LOCAL_CONFIG" ]; then
    echo "Using local overrides from $LOCAL_CONFIG"
    ICE_CONFIG_FILE="$LOCAL_CONFIG"
fi

if [ ! -f "$ICE_CONFIG_FILE" ]; then
    echo "ERROR: Config file '$ICE_CONFIG_FILE' not found."
    exit 1
fi

ICE_SECTION=$(sed -n '/^\[ice\]/,/^\s*\[/p' "$ICE_CONFIG_FILE")
[ -z "$ICE_SECTION" ] && ICE_SECTION=$(sed -n '/^\[ice\]/,$p' "$ICE_CONFIG_FILE")

ICE_ACCOUNT=$(echo   "$ICE_SECTION" | grep -E "^\s*account\s*="   | sed 's/.*=\s*//;s/[" ]//g;s/#.*//')
ICE_PARTITION=$(echo "$ICE_SECTION" | grep -E "^\s*partition\s*="  | sed 's/.*=\s*//;s/[" ]//g;s/#.*//')
ICE_WALLTIME=$(echo  "$ICE_SECTION" | grep -E "^\s*walltime\s*="   | sed 's/.*=\s*//;s/[" ]//g;s/#.*//')

[ -z "$ICE_ACCOUNT"  ] && ICE_ACCOUNT="coa"
[ -z "$ICE_WALLTIME" ] && ICE_WALLTIME="08:00:00"

GPU_TYPE_UPPER=$(echo "$GPU_TYPE" | tr '[:lower:]' '[:upper:]')
SLURM_GPU="${GPU_TYPE_UPPER}:${NUM_GPUS}"

echo "=========================================="
echo " Deploying HybridFNO on ICE"
echo " Account:   $ICE_ACCOUNT"
echo " Partition: ${ICE_PARTITION:-default}"
echo " GPU:       $SLURM_GPU"
echo " Walltime:  $ICE_WALLTIME"
echo " Config:    $CONFIG_FILE"
echo " Fresh:     ${FRESH_TRAIN:-no}"
echo "=========================================="

mkdir -p logs

SBATCH_CMD="sbatch \
    --gres=gpu:$SLURM_GPU \
    --account=$ICE_ACCOUNT \
    --time=$ICE_WALLTIME \
    --export=NUM_GPUS=$NUM_GPUS,CONFIG_FILE=$ICE_CONFIG_FILE,FRESH_TRAIN=$FRESH_TRAIN,RESET_PATIENCE=$RESET_PATIENCE"

[ -n "$ICE_PARTITION" ] && SBATCH_CMD="$SBATCH_CMD --partition=$ICE_PARTITION"

SBATCH_CMD="$SBATCH_CMD slurm/train_hybrid.sbatch"

$SBATCH_CMD

echo "Check status: squeue -u \$USER"
