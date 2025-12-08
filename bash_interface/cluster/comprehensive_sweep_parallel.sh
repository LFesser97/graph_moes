#!/bin/bash
#SBATCH --job-name=comprehensive_sweep
#SBATCH --array=1-110             # Total experiments: 50 single layer + 60 MoE = 110
#SBATCH --ntasks=1
#SBATCH --time=48:00:00           # Long time for comprehensive sweep
#SBATCH --mem=64GB               # Sufficient memory
#SBATCH --output=logs_comprehensive/comprehensive_sweep_%A_%a.log  # %A = array job ID, %a = task ID
#SBATCH --partition=mweber_gpu
#SBATCH --gpus=1

# WandB Environment Setup
echo "🚀 Setting up WandB environment for Comprehensive Graph MoE experiments..."

export WANDB_API_KEY="ea7c6eeb5a095b531ef60cc784bfeb87d47ea0b0"
export WANDB_ENTITY="weber-geoml-harvard-university"
export WANDB_PROJECT="MOE"
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

# Set environment path and activate moe environment
export CONDA_ENVS_PATH=/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/conda/envs
source activate moe

# Navigate to project directory
cd /n/holylabs/mweber_lab/Everyone/rpellegrin/graph_moes

# Quick verification
python -c "import numpy, pandas, torch; print('✅ Core packages available')" || {
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
source /n/holylabs/mweber_lab/Everyone/rpellegrin/graph_moes/bash_interface/cluster/hyperparams_lookup.sh

# Define datasets - each will use optimal hyperparameters from research paper
datasets=(enzymes proteins mutag imdb collab reddit mnist cifar pattern cluster)

# Calculate which experiment this array task should run
task_id=${SLURM_ARRAY_TASK_ID:-1}

# Total single layer experiments: 5 layer types × 10 datasets = 50
# Total MoE experiments: 6 combinations × 10 datasets = 60
# Total: 110 experiments

if [ "$task_id" -le 50 ]; then
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
    moe_id=$((task_id - 51))
    
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
