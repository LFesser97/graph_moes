#!/bin/bash
# ============================================================================
# Comprehensive Graph MoE Sweep - Parallel Array Job Version
# ============================================================================
# This script runs a comprehensive hyperparameter sweep for graph neural network
# experiments using SLURM array jobs for parallel execution. It tests both single
# layer architectures (GCN, GIN, SAGE, MLP, Unitary) and MoE (Mixture of Experts)
# combinations across multiple datasets.
#
# The script uses optimal hyperparameters from research papers for each dataset
# and model combination, loaded from hyperparams_lookup.sh.
#
# Total experiments: 154
#   - 70 single layer experiments: 5 layer types × 14 datasets (10 original + 4 GraphBench)
#   - 84 MoE experiments: 6 layer combinations × 14 datasets
#
# Usage: sbatch comprehensive_sweep_parallel.sh
# ============================================================================

#SBATCH --job-name=comprehensive_sweep
#SBATCH --array=1-154             # Total experiments: 70 single layer + 84 MoE = 154
#SBATCH --ntasks=1
#SBATCH --time=48:00:00           # Long time for comprehensive sweep
#SBATCH --mem=64GB               # Sufficient memory
#SBATCH --output=logs_comprehensive/Parrallel_comprehensive_sweep_%A_%a.log  # %A = array job ID, %a = task ID
#SBATCH --partition=mweber_gpu
#SBATCH --gpus=1

# WandB Environment Setup
echo "🚀 Setting up WandB environment for Comprehensive Graph MoE experiments..."

export WANDB_API_KEY="ea7c6eeb5a095b531ef60cc784bfeb87d47ea0b0"
export WANDB_ENTITY="weber-geoml-harvard-university"
export WANDB_PROJECT="MOE_DECEMBER25"
export WANDB_DIR="./wandb"
export WANDB_CACHE_DIR="./wandb/.cache"

mkdir -p ./wandb logs

echo "✅ WandB environment configured"
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
echo "🎉 WandB setup complete!"

# Function to log messages with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Task $SLURM_ARRAY_TASK_ID] $1"
}

log_message "Starting Comprehensive MoE Sweep Task $SLURM_ARRAY_TASK_ID"

# Set environment path
export CONDA_ENVS_PATH=/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/conda/envs
export CONDA_PKGS_DIRS=/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/conda/pkgs

# Load Python module (provides mamba/conda infrastructure)
log_message "📦 Loading Python module..."
module load python/3.10.12-fasrc01 || {
    log_message "⚠️  Failed to load python module, trying alternative..."
    # Try alternative module name
    module load python/3.10-fasrc01 || {
        log_message "❌ Failed to load Python module"
        log_message "   Available modules:"
        module avail python 2>&1 | head -5
        exit 1
    }
}

# Verify mamba is available
if ! command -v mamba &> /dev/null && ! command -v conda &> /dev/null; then
    log_message "❌ Neither mamba nor conda available after loading Python module"
    exit 1
fi

log_message "✅ Python module loaded: $(which python)"

# Initialize conda/mamba (required for source activate to work properly)
log_message "🔧 Initializing conda/mamba..."
if [ -f "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    log_message "✅ Conda initialized from conda info --base"
elif [ -f "/n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh" ]; then
    source "/n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh"
    log_message "✅ Conda initialized from Mambaforge path"
else
    log_message "⚠️  Could not find conda.sh, activation may not work properly"
fi

# Activate environment
log_message "🔧 Activating moe environment..."
log_message "   CONDA_ENVS_PATH: $CONDA_ENVS_PATH"
log_message "   Python before activation: $(which python)"

# Try activation methods
activation_success=false

# Method 1: source activate
if source activate moe 2>/dev/null; then
    python_path=$(which python)
    if [[ "$python_path" == *"moe"* ]]; then
        log_message "✅ Activated using 'source activate'"
        log_message "   Python after activation: $python_path"
        activation_success=true
    else
        log_message "⚠️  'source activate' ran but Python path unchanged: $python_path"
    fi
fi

# Method 2: conda activate (if first method didn't work)
if [ "$activation_success" = false ] && command -v conda &> /dev/null; then
    if conda activate moe 2>/dev/null; then
        python_path=$(which python)
        if [[ "$python_path" == *"moe"* ]]; then
            log_message "✅ Activated using 'conda activate'"
            log_message "   Python after activation: $python_path"
            activation_success=true
        else
            log_message "⚠️  'conda activate' ran but Python path unchanged: $python_path"
        fi
    fi
fi

# Method 3: Direct path activation (if previous methods didn't work)
if [ "$activation_success" = false ]; then
    if [ -f "$CONDA_ENVS_PATH/moe/bin/activate" ]; then
        log_message "   Trying direct path activation..."
        source "$CONDA_ENVS_PATH/moe/bin/activate"
        python_path=$(which python)
        if [[ "$python_path" == *"moe"* ]]; then
            log_message "✅ Activated using direct path"
            log_message "   Python after activation: $python_path"
            activation_success=true
        else
            log_message "⚠️  Direct path activation ran but Python path unchanged: $python_path"
        fi
    fi
fi

# Method 4: Manually set PATH if activation didn't work
if [ "$activation_success" = false ]; then
    if [ -d "$CONDA_ENVS_PATH/moe/bin" ] && [ -f "$CONDA_ENVS_PATH/moe/bin/python" ]; then
        log_message "   Trying manual PATH setup..."
        # Prepend moe bin to PATH
        export PATH="$CONDA_ENVS_PATH/moe/bin:$PATH"
        # Also set CONDA_DEFAULT_ENV for compatibility
        export CONDA_DEFAULT_ENV=moe
        python_path=$(which python)
        if [[ "$python_path" == *"moe"* ]]; then
            log_message "✅ Activated using manual PATH setup"
            log_message "   Python after activation: $python_path"
            activation_success=true
        else
            log_message "⚠️  Manual PATH setup didn't work, Python still: $python_path"
            log_message "   Checking if moe/bin/python exists..."
            ls -la "$CONDA_ENVS_PATH/moe/bin/python" 2>&1 || log_message "   ❌ moe/bin/python does not exist!"
        fi
    else
        log_message "⚠️  moe/bin directory or python not found at: $CONDA_ENVS_PATH/moe/bin"
    fi
fi

# Final verification
if [ "$activation_success" = false ]; then
    log_message "❌ Failed to activate moe environment"
    log_message "   Current Python: $(which python)"
    log_message "   Expected path should contain: moe"
    log_message "   CONDA_ENVS_PATH: $CONDA_ENVS_PATH"
    if [ -d "$CONDA_ENVS_PATH" ]; then
        log_message "   Available environments:"
        ls -la "$CONDA_ENVS_PATH/" 2>&1 | head -10
    else
        log_message "   CONDA_ENVS_PATH directory does not exist!"
    fi
    exit 1
fi

# Double-check Python is from moe environment
python_path=$(which python)
if [[ "$python_path" != *"moe"* ]]; then
    log_message "❌ CRITICAL: Python not from moe environment after activation!"
    log_message "   Python path: $python_path"
    log_message "   This will cause import errors!"
    exit 1
fi

log_message "✅ Verified moe environment active: $python_path"

# Navigate to project directory
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/graph_moes || {
    log_message "❌ Failed to navigate to project directory"
    exit 1
}

log_message "📁 Project directory: $(pwd)"

# Install required packages that might be missing
log_message "📦 Checking and installing required packages..."
if ! python -c "import graphbench" 2>/dev/null; then
    log_message "   Installing graphbench-lib..."
    pip install graphbench-lib --quiet || {
        log_message "⚠️  Failed to install graphbench-lib, continuing anyway..."
    }
else
    log_message "   ✅ graphbench-lib already installed"
fi

# Install project in development mode (if not already installed)
# Use --no-deps to avoid rebuilding torch-cluster and other compiled packages
if ! python -c "import graph_moes" 2>/dev/null; then
    log_message "📦 Installing graph_moes project (without dependencies)..."
    # Try installing without dependencies first (dependencies should already be in environment)
    if pip install -e . --no-deps --quiet 2>/dev/null; then
        log_message "✅ Project installed (no-deps mode)"
    # If that fails, try adding src to PYTHONPATH as fallback
    elif [ -d "src" ]; then
        log_message "⚠️  pip install failed, adding src to PYTHONPATH..."
        export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
        if python -c "import graph_moes" 2>/dev/null; then
            log_message "✅ Project accessible via PYTHONPATH"
        else
            log_message "❌ Failed to make graph_moes importable"
            exit 1
        fi
    else
        log_message "❌ Failed to install graph_moes project and src directory not found"
        exit 1
    fi
else
    log_message "✅ graph_moes already installed"
fi

# Quick verification
python -c "import numpy, pandas, torch, graph_moes; print('✅ Core packages available')" || {
    log_message "❌ Core packages not available"
    exit 1
}

# Define all layer type combinations
declare -a single_layer_types=(
    "GCN"
    "GIN" 
    "SAGE"
    "MLP"
    "Unitary"
)

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

# Define datasets - each will use optimal hyperparameters from research paper
# Includes GraphBench datasets with "graphbench_" prefix
datasets=(enzymes proteins mutag imdb collab reddit mnist cifar pattern cluster graphbench_socialnetwork graphbench_co graphbench_sat graphbench_weather)

# Calculate which experiment this array task should run
task_id=${SLURM_ARRAY_TASK_ID:-1}

# Total single layer experiments: 5 layer types × 14 datasets = 70
# Total MoE experiments: 6 combinations × 14 datasets = 84
# Total: 154 experiments

if [ "$task_id" -le 70 ]; then
    # Single layer experiment
    experiment_type="single"
    adjusted_id=$((task_id - 1))
    
    # Calculate dataset and layer type
    num_datasets=${#datasets[@]}
    dataset_idx=$((adjusted_id % num_datasets))
    layer_idx=$((adjusted_id / num_datasets))
    
    dataset=${datasets[$dataset_idx]}
    layer_type=${single_layer_types[$layer_idx]}
    
    log_message "🧪 Single Layer Experiment $task_id: ${dataset}_${layer_type}"
    
    # Get optimal hyperparameters
    get_hyperparams "$dataset" "$layer_type"
    
    # Extract hyperparameters
    learning_rate=$HYPERPARAM_LEARNING_RATE
    hidden_dim=$HYPERPARAM_HIDDEN_DIM
    num_layer=$HYPERPARAM_NUM_LAYERS
    dropout=$HYPERPARAM_DROPOUT
    patience=$HYPERPARAM_PATIENCE
    
    wandb_run_name="${dataset}_${layer_type}_L${num_layer}_H${hidden_dim}_lr${learning_rate}_d${dropout}_task${task_id}"
    
    # Run single layer experiment
    python scripts/run_graph_classification.py \
        --num_trials 5 \
        --dataset "$dataset" \
        --layer_type "$layer_type" \
        --learning_rate "$learning_rate" \
        --hidden_dim "$hidden_dim" \
        --num_layers "$num_layer" \
        --dropout "$dropout" \
        --patience "$patience" \
        --wandb_enabled \
        --wandb_name "$wandb_run_name" \
        --wandb_tags '["cluster", "comprehensive", "single_layer", "research_hyperparams"]'

else
    # MoE experiment
    experiment_type="moe"
    moe_id=$((task_id - 71))
    
    # Calculate dataset and MoE combination
    num_datasets=${#datasets[@]}
    dataset_idx=$((moe_id % num_datasets))
    combo_idx=$((moe_id / num_datasets))
    
    dataset=${datasets[$dataset_idx]}
    layer_combo=${moe_combinations[$combo_idx]}
    
    log_message "🧪 MoE Experiment $task_id: ${dataset}_MoE_${layer_combo}"
    
    # For MoE, use the first layer type in combination as base for hyperparameters
    first_layer=$(echo "$layer_combo" | grep -o '"[^"]*"' | head -1 | tr -d '"')
    get_hyperparams "$dataset" "$first_layer"
    
    # Extract hyperparameters
    learning_rate=$HYPERPARAM_LEARNING_RATE
    hidden_dim=$HYPERPARAM_HIDDEN_DIM
    num_layer=$HYPERPARAM_NUM_LAYERS
    dropout=$HYPERPARAM_DROPOUT
    patience=$HYPERPARAM_PATIENCE
    
    # Create a clean name from layer combination
    clean_combo=$(echo "$layer_combo" | tr -d '[]",' | tr ' ' '_')
    wandb_run_name="${dataset}_MoE_${clean_combo}_L${num_layer}_H${hidden_dim}_lr${learning_rate}_d${dropout}_task${task_id}"
    
    # Run MoE experiment
    python scripts/run_graph_classification.py \
        --num_trials 5 \
        --dataset "$dataset" \
        --layer_types "$layer_combo" \
        --learning_rate "$learning_rate" \
        --hidden_dim "$hidden_dim" \
        --num_layers "$num_layer" \
        --dropout "$dropout" \
        --patience "$patience" \
        --wandb_enabled \
        --wandb_name "$wandb_run_name" \
        --wandb_tags '["cluster", "comprehensive", "moe", "research_hyperparams"]'
fi

# Check exit status
if [ $? -eq 0 ]; then
    log_message "✅ Task $task_id ($experiment_type) completed successfully"
else
    log_message "❌ Task $task_id ($experiment_type) failed with exit code $?"
    exit 1
fi

log_message "🎉 Task $task_id completed!"
