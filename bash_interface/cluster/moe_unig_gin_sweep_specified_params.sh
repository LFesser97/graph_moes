#!/bin/bash
# ============================================================================
# MoE GIN+Unitary Hyperparameter Sweep
# ============================================================================
# This script runs a hyperparameter sweep for MoE (Mixture of Experts) experiments
# combining GIN and Unitary layer types. It uses SLURM array jobs to test different
# hyperparameter combinations in parallel.
#
# The script sweeps over:
#   - Datasets: proteins, mutag (2 datasets)
#   - Learning rates: 0.001, 0.0001 (2 values)
#   - Hidden dimensions: 64, 128 (2 values)
#   - Number of layers: 4, 6 (2 values)
#   - Dropout rates: 0.0, 0.1, 0.2 (3 values)
#
# Total combinations: 2 × 2 × 2 × 2 × 3 = 48 experiments
# Each experiment runs 5 trials.
#
# Usage: sbatch moe_unig_gin_sweep_specified_params.sh
# ============================================================================

#SBATCH --job-name=moe_uni_gin_array
#SBATCH --array=1-48              # Total combinations: 2 datasets × 2 lr × 2 hidden × 2 layers × 3 dropout = 48
#SBATCH --ntasks=1
#SBATCH --time=8:00:00
#SBATCH --mem=64GB
#SBATCH --output=logs_uni_gin/moe_uni_gin_%A_%a.log  # %A = array job ID, %a = task ID
#SBATCH --partition=mweber_gpu
#SBATCH --gpus=1

# WandB Environment Setup for Graph MoE Experiments
echo "🚀 Setting up WandB environment for Graph MoE experiments..."

# Set WandB environment variables
export WANDB_API_KEY="ea7c6eeb5a095b531ef60cc784bfeb87d47ea0b0"
export WANDB_ENTITY="weber-geoml-harvard-university"
export WANDB_PROJECT="MOE_DECEMBER25"

# Optional: Set other WandB configurations
export WANDB_DIR="./wandb"
export WANDB_CACHE_DIR="./wandb/.cache"

# Create wandb directory if it doesn't exist
mkdir -p ./wandb

echo "✅ WandB environment configured:"
echo "   Entity: $WANDB_ENTITY"
echo "   Project: $WANDB_PROJECT"
echo "   API Key: ${WANDB_API_KEY:0:10}..."
echo "   Directory: $WANDB_DIR"

# Install wandb if not already installed
if ! python -c "import wandb" &> /dev/null; then
    echo "📦 Installing wandb..."
    pip install wandb
else
    echo "✅ wandb already installed"
fi

echo "🎉 WandB setup complete!"

# Create logs directory
mkdir -p logs

# Function to log messages with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Task $SLURM_ARRAY_TASK_ID] $1"
}

log_message "Starting MoE GIN+Unitary task $SLURM_ARRAY_TASK_ID"

# Set environment path and activate moe environment
export CONDA_ENVS_PATH=/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/conda/envs
source activate moe

# Force check that we're in the right environment
if [[ "$(which python)" != *"moe"* ]]; then
    log_message "❌ Python not from moe environment: $(which python)"
    exit 1
fi

# Check if environment activation was successful
if [[ "$CONDA_DEFAULT_ENV" == "moe" ]] || [[ "$CONDA_DEFAULT_ENV" == *"moe"* ]]; then
    log_message "✅ Successfully activated moe mamba environment: $CONDA_DEFAULT_ENV"
    log_message "🐍 Python path: $(which python)"
    log_message "📦 Conda environment: $CONDA_DEFAULT_ENV"
else
    log_message "❌ Failed to activate moe environment! Current env: $CONDA_DEFAULT_ENV"
    log_message "🚨 This will likely cause import errors"
    exit 1
fi

# Navigate to project directory
cd /n/holylabs/mweber_lab/Everyone/rpellegrin/graph_moes

# Verify we're in the right directory
if [[ -f "scripts/run_graph_classification.py" ]]; then
    log_message "✅ Successfully navigated to project directory: $(pwd)"
else
    log_message "❌ Failed to find project files in: $(pwd)"
    log_message "🔍 Looking for scripts/run_graph_classification.py"
    exit 1
fi

# Quick verification that packages work
log_message "🔍 Quick package verification..."
python -c "import numpy, pandas, torch; print('✅ Core packages available')" || {
    log_message "❌ Core packages not available - recreate mamba environment"
    exit 1
}

# Define hyperparameter combinations
datasets=(enzymes proteins mutag imdb collab reddit mnist cifar pattern cluster)
# # All available datasets from scripts/run_graph_classification.py
# datasets=(mutag enzymes proteins imdb collab reddit mnist cifar pattern cluster pascalvoc coco molhiv molpcba)

learning_rates=(0.001 0.0001)
hidden_dims=(64 128)
num_layers_list=(4 6)
dropouts=(0.0 0.1 0.2)

# Calculate which combination this task should run
# Total combinations per dataset: 2 × 2 × 2 × 3 = 24
# Task 1-24: proteins, Task 25-48: mutag

task_id=$SLURM_ARRAY_TASK_ID
if [ $task_id -le 24 ]; then
    dataset="proteins"
    combo_id=$((task_id - 1))
else
    dataset="mutag"
    combo_id=$((task_id - 25))
fi

# Convert combo_id to specific hyperparameters
total_lr=${#learning_rates[@]}
total_hd=${#hidden_dims[@]}
total_nl=${#num_layers_list[@]}
total_do=${#dropouts[@]}

# Calculate indices using modular arithmetic
do_idx=$((combo_id % total_do))
combo_id=$((combo_id / total_do))
nl_idx=$((combo_id % total_nl))
combo_id=$((combo_id / total_nl))
hd_idx=$((combo_id % total_hd))
lr_idx=$((combo_id / total_hd))

# Get actual values
learning_rate=${learning_rates[$lr_idx]}
hidden_dim=${hidden_dims[$hd_idx]}
num_layer=${num_layers_list[$nl_idx]}
dropout=${dropouts[$do_idx]}

log_message "Configuration: dataset=$dataset, lr=$learning_rate, hidden_dim=$hidden_dim, num_layers=$num_layer, dropout=$dropout"

# Generate wandb run name for this specific task
wandb_run_name="${dataset}_GIN_Unitary_L${num_layer}_H${hidden_dim}_lr${learning_rate}_d${dropout}_task${task_id}"

log_message "WandB run name: $wandb_run_name"

# Run the experiment with wandb enabled
python scripts/run_graph_classification.py \
    --num_trials 5 \
    --dataset "$dataset" \
    --learning_rate "$learning_rate" \
    --hidden_dim "$hidden_dim" \
    --num_layers "$num_layer" \
    --dropout "$dropout" \
    --patience 50 \
    --layer_types '["GIN", "Unitary"]' \
    --wandb_enabled \
    --wandb_name "$wandb_run_name" \
    --wandb_tags '["cluster", "sweep", "gin_unitary"]'

# Check exit status
if [ $? -eq 0 ]; then
    log_message "✅ Task completed successfully"
else
    log_message "❌ Task failed with exit code $?"
    exit 1
fi