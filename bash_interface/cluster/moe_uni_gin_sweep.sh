#!/bin/bash
# ============================================================================
# MoE GIN+Unitary Sweep with Research-Based Hyperparameters
# ============================================================================
# This script runs MoE (Mixture of Experts) experiments combining GIN and Unitary
# layer types across multiple datasets using optimal hyperparameters from research
# papers. It uses SLURM array jobs to run experiments in parallel.
#
# Each array task runs one dataset with optimal hyperparameters loaded from
# hyperparams_lookup.sh. The hyperparameters are based on GIN settings for the
# given dataset.
#
# Configuration:
#   - Layer types: ["GIN", "Unitary"] (MoE combination)
#   - Number of trials: 5 per experiment
#   - Hyperparameters: Research-based optimal values per dataset
#   - Datasets: enzymes, proteins, mutag, imdb, collab, reddit, mnist, cifar, pattern
#
# Usage: sbatch moe_uni_gin_sweep.sh
# ============================================================================

#SBATCH --job-name=moe_uni_gin_array
#SBATCH --array=1-10              # Total datasets: 10 datasets with optimal hyperparameters each
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
export WANDB_PROJECT="MOE"

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
mkdir -p logs logs_uni_gin

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

# Load hyperparameter lookup function
source /n/holylabs/mweber_lab/Everyone/rpellegrin/graph_moes/bash_interface/cluster/hyperparams_lookup.sh

# Define datasets to run experiments on
datasets=(enzymes proteins mutag imdb collab reddit mnist cifar pattern)
# # All available datasets from scripts/run_graph_classification.py
# datasets=(mutag enzymes proteins imdb collab reddit mnist cifar pattern cluster pascalvoc coco molhiv molpcba)

# Calculate which dataset this task should run
# Each dataset gets optimal hyperparameters from research paper
task_id=${SLURM_ARRAY_TASK_ID:-1}
total_datasets=${#datasets[@]}
dataset_idx=$((($task_id - 1) % $total_datasets))
dataset=${datasets[$dataset_idx]}

# Get optimal hyperparameters for this dataset and model combination
# For MoE with GIN+Unitary, use GIN+ as base
get_hyperparams "$dataset" "GIN"

# Extract hyperparameters from the lookup function
learning_rate=$HYPERPARAM_LEARNING_RATE
hidden_dim=$HYPERPARAM_HIDDEN_DIM
num_layer=$HYPERPARAM_NUM_LAYERS
dropout=$HYPERPARAM_DROPOUT
batch_size=$HYPERPARAM_BATCH_SIZE
epochs=$HYPERPARAM_EPOCHS
patience=$HYPERPARAM_PATIENCE

log_message "Configuration: dataset=$dataset, lr=$learning_rate, hidden_dim=$hidden_dim, num_layers=$num_layer, dropout=$dropout"

# Generate wandb run name for this specific task
wandb_run_name="${dataset}_GIN_Unitary_L${num_layer}_H${hidden_dim}_lr${learning_rate}_d${dropout}_task${task_id}"

log_message "WandB run name: $wandb_run_name"

# Run the experiment with wandb enabled using research-based hyperparameters
python scripts/run_graph_classification.py \
    --num_trials 5 \
    --dataset "$dataset" \
    --learning_rate "$learning_rate" \
    --hidden_dim "$hidden_dim" \
    --num_layers "$num_layer" \
    --dropout "$dropout" \
    --patience "$patience" \
    --layer_types '["GIN", "Unitary"]' \
    --wandb_enabled \
    --wandb_name "$wandb_run_name" \
    --wandb_tags '["cluster", "sweep", "gin_unitary", "research_hyperparams"]'

# Check exit status
if [ $? -eq 0 ]; then
    log_message "✅ Task completed successfully"
else
    log_message "❌ Task failed with exit code $?"
    exit 1
fi
