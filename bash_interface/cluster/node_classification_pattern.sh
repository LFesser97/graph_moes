#!/bin/bash
# ============================================================================
# Node Classification Experiments on PATTERN Dataset
# ============================================================================
# This script runs node classification experiments on the PATTERN dataset.
# Unlike graph classification, node classification predicts labels for individual
# nodes within graphs rather than classifying entire graphs.
#
# PATTERN dataset: 10,000 graphs, each with ~108 nodes, binary node labels
# Task: Predict binary labels (0/1) for each node in synthetic patterns
#
# Usage: sbatch node_classification_pattern.sh
# ============================================================================

#SBATCH --job-name=node_classification_pattern
#SBATCH --array=1-40             # 5 layer types × 8 combinations = 40 experiments
#SBATCH --ntasks=1
#SBATCH --time=24:00:00          # Moderate time for node classification
#SBATCH --mem=32GB              # Sufficient memory for node classification
#SBATCH --output=logs_node_classification/Pattern_node_cls_%A_%a.log
#SBATCH --partition=mweber_gpu
#SBATCH --gpus=1

# WandB Environment Setup
echo "🚀 Setting up WandB environment for Node Classification on PATTERN..."

export WANDB_API_KEY="ea7c6eeb5a095b531ef60cc784bfeb87d47ea0b0"
export WANDB_ENTITY="weber-geoml-harvard-university"
export WANDB_PROJECT="MOE_PATTERN_Node_Classification"
export WANDB_DIR="./wandb"
export WANDB_CACHE_DIR="./wandb/.cache"

mkdir -p ./wandb logs_node_classification

# Disable user site-packages to prevent conflicts
export PYTHONNOUSERSITE=1

# Load CUDA modules if available
if command -v module &> /dev/null; then
    echo "📦 Loading CUDA modules..."
    if module load cuda/12.9.1-fasrc01 2>/dev/null; then
        echo "   ✅ Loaded cuda/12.9.1-fasrc01"
    elif module load cuda 2>/dev/null; then
        echo "   ✅ Loaded cuda (default version)"
    else
        echo "   ⚠️  CUDA module not found, continuing..."
    fi

    if module load cudnn/9.10.2.21_cuda12-fasrc01 2>/dev/null; then
        echo "   ✅ Loaded cudnn/9.10.2.21_cuda12-fasrc01"
    elif module load cudnn 2>/dev/null; then
        echo "   ✅ Loaded cudnn (default version)"
    else
        echo "   ⚠️  cuDNN module not found, continuing..."
    fi

    # Set LD_LIBRARY_PATH
    if [ -n "$CUDA_HOME" ]; then
        export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"
    elif [ -n "$CUDA_ROOT" ]; then
        export LD_LIBRARY_PATH="${CUDA_ROOT}/lib64:${CUDA_ROOT}/lib:${CUDA_ROOT}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"
    fi
fi

echo "✅ WandB environment configured"
echo "   Entity: $WANDB_ENTITY"
echo "   Project: $WANDB_PROJECT"

# Install wandb if not available
if ! python -c "import wandb" &> /dev/null; then
    echo "📦 Installing wandb..."
    pip install wandb
else
    echo "✅ wandb already installed"
fi

echo "🎉 WandB setup complete!"

# Function to log messages with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Task $SLURM_ARRAY_TASK_ID] $1"
}

log_message "Starting PATTERN Node Classification Task $SLURM_ARRAY_TASK_ID"

# Set environment paths
export CONDA_ENVS_PATH=/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/conda/envs
export CONDA_PKGS_DIRS=/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/conda/pkgs

# Load Python module
log_message "📦 Loading Python module..."
module load python/3.10.12-fasrc01 || {
    log_message "⚠️  Failed to load python module, trying alternative..."
    module load python/3.10-fasrc01 || {
        log_message "❌ Failed to load Python module"
        exit 1
    }
}

# Activate environment
ENV_NAME="moe_fresh"
log_message "🔧 Activating $ENV_NAME environment..."

# Use manual PATH setup
activation_success=false
if [ -d "$CONDA_ENVS_PATH/$ENV_NAME/bin" ] && [ -f "$CONDA_ENVS_PATH/$ENV_NAME/bin/python" ]; then
    export PATH="$CONDA_ENVS_PATH/$ENV_NAME/bin:$PATH"
    export CONDA_DEFAULT_ENV=$ENV_NAME
    export PYTHONNOUSERSITE=1
    python_path=$(which python)
    if [[ "$python_path" == *"$ENV_NAME"* ]]; then
        log_message "✅ Environment activated"
        activation_success=true
    else
        log_message "⚠️  PATH setup didn't work"
    fi
else
    log_message "⚠️  $ENV_NAME environment not found"
fi

if [ "$activation_success" = false ]; then
    log_message "❌ Failed to activate environment"
    exit 1
fi

# Verify Python environment
python_path=$(which python)
if [[ "$python_path" != *"$ENV_NAME"* ]]; then
    log_message "❌ Python not from correct environment"
    exit 1
fi

log_message "✅ Environment active: $python_path"

# Navigate to project directory
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/graph_moes || {
    log_message "❌ Failed to navigate to project directory"
    exit 1
}

log_message "📁 Project directory: $(pwd)"

# Set PYTHONPATH
export PYTHONPATH="$(pwd):$(pwd)/src:${PYTHONPATH}"

# Install project if needed
if ! python -c "import graph_moes" 2>/dev/null; then
    log_message "📦 Installing graph_moes project..."
    pip install -e . --no-deps --quiet 2>&1 || {
        export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
    }
fi

# Quick verification
python -c "import numpy, pandas, torch, graph_moes; print('✅ Core packages available')" || {
    log_message "❌ Core packages not available"
    exit 1
}

# Define layer types for node classification
declare -a layer_types=(
    "GCN"
    "GIN"
    "SAGE"
    "MLP"
    "Unitary"
)

# Define MoE combinations for node classification
declare -a moe_combinations=(
    '["GCN", "GIN"]'
    '["GCN", "SAGE"]'
    '["GCN", "Unitary"]'
    '["GIN", "SAGE"]'
    '["GIN", "Unitary"]'
    '["SAGE", "Unitary"]'
)

# Load hyperparameter lookup function
source /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/graph_moes/bash_interface/cluster/hyperparams_lookup.sh

# Calculate which experiment to run
task_id=${SLURM_ARRAY_TASK_ID:-1}

# Total experiments: 5 single layer + 6 MoE = 11 experiments
if [ "$task_id" -le 5 ]; then
    # Single layer experiment
    experiment_type="single"
    layer_idx=$((task_id - 1))
    layer_type=${layer_types[$layer_idx]}

    log_message "🧪 PATTERN Node Classification - Single Layer: $layer_type"

    # Get hyperparameters for PATTERN (binary node classification)
    get_hyperparams "pattern" "$layer_type"

    # Extract hyperparameters
    learning_rate=$HYPERPARAM_LEARNING_RATE
    hidden_dim=$HYPERPARAM_HIDDEN_DIM
    num_layer=$HYPERPARAM_NUM_LAYERS
    dropout=$HYPERPARAM_DROPOUT
    patience=$HYPERPARAM_PATIENCE

    wandb_run_name="PATTERN_node_cls_${layer_type}_L${num_layer}_H${hidden_dim}_lr${learning_rate}_d${dropout}_task${task_id}"

    # Run single layer node classification experiment
    python scripts/run_graph_classification.py \
        --dataset "pattern" \
        --layer_type "$layer_type" \
        --learning_rate "$learning_rate" \
        --hidden_dim "$hidden_dim" \
        --num_layers "$num_layer" \
        --dropout "$dropout" \
        --patience "$patience" \
        --wandb_enabled \
        --wandb_name "$wandb_run_name" \
        --wandb_tags '["cluster", "node_classification", "pattern", "single_layer"]'

else
    # MoE experiment
    experiment_type="moe"
    moe_idx=$((task_id - 6))
    layer_combo=${moe_combinations[$moe_idx]}

    log_message "🧪 PATTERN Node Classification - MoE: $layer_combo"

    # For MoE, use first layer type for hyperparameters
    first_layer=$(echo "$layer_combo" | grep -o '"[^"]*"' | head -1 | tr -d '"')
    get_hyperparams "pattern" "$first_layer"

    # Extract hyperparameters
    learning_rate=$HYPERPARAM_LEARNING_RATE
    hidden_dim=$HYPERPARAM_HIDDEN_DIM
    num_layer=$HYPERPARAM_NUM_LAYERS
    dropout=$HYPERPARAM_DROPOUT
    patience=$HYPERPARAM_PATIENCE

    # Create clean name for MoE combination
    clean_combo=$(echo "$layer_combo" | tr -d '[]",' | tr ' ' '_')
    wandb_run_name="PATTERN_node_cls_MoE_${clean_combo}_L${num_layer}_H${hidden_dim}_lr${learning_rate}_d${dropout}_task${task_id}"

    # Run MoE node classification experiment
    python scripts/run_graph_classification.py \
        --dataset "pattern" \
        --layer_types "$layer_combo" \
        --learning_rate "$learning_rate" \
        --hidden_dim "$hidden_dim" \
        --num_layers "$num_layer" \
        --dropout "$dropout" \
        --patience "$patience" \
        --wandb_enabled \
        --wandb_name "$wandb_run_name" \
        --wandb_tags '["cluster", "node_classification", "pattern", "moe"]'
fi

# Check exit status
if [ $? -eq 0 ]; then
    log_message "✅ Task $task_id ($experiment_type) completed successfully"
else
    log_message "❌ Task $task_id ($experiment_type) failed with exit code $?"
    exit 1
fi

log_message "🎉 Task $task_id completed!"
