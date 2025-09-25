#!/bin/bash
#SBATCH --job-name=comprehensive_sweep
#SBATCH --ntasks=1
#SBATCH --time=48:00:00           # Long time for comprehensive sweep
#SBATCH --mem=64GB               # Sufficient memory
#SBATCH --output=logs_comprehensiv/comprehensive_sweep_%j.log
#SBATCH --partition=mweber_gpu
#SBATCH --gpus=1

# WandB Environment Setup
echo "🚀 Setting up WandB environment for Comprehensive Graph MoE experiments..."

export WANDB_API_KEY="ea7c6eeb5a095b531ef60cc784bfeb87d47ea0b0"
export WANDB_ENTITY="weber-geoml-harvard-university"
export WANDB_PROJECT="MOE"
export WANDB_DIR="./wandb"
export WANDB_CACHE_DIR="./wandb/.cache"

mkdir -p ./wandb logs logs_comprehensiv

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
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log_message "Starting Comprehensive MoE Sweep"

# Set environment path and activate moe environment
export CONDA_ENVS_PATH=/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/conda/envs
source activate moe

# Navigate to project directory
cd /n/holylabs/mweber_lab/Everyone/rpellegrin/graph_moes



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

# Experiment counter
experiment_id=1

log_message "🔬 Starting Single Layer Experiments..."

# Single Layer Experiments
for layer_type in "${single_layer_types[@]}"; do
    for dataset in "${datasets[@]}"; do
        # Get optimal hyperparameters for this dataset and layer type
        get_hyperparams "$dataset" "$layer_type"
        
        # Extract hyperparameters
        learning_rate=$HYPERPARAM_LEARNING_RATE
        hidden_dim=$HYPERPARAM_HIDDEN_DIM
        num_layer=$HYPERPARAM_NUM_LAYERS
        dropout=$HYPERPARAM_DROPOUT
        patience=$HYPERPARAM_PATIENCE
        
        wandb_run_name="${dataset}_${layer_type}_L${num_layer}_H${hidden_dim}_lr${learning_rate}_d${dropout}_exp${experiment_id}"
        
        log_message "🧪 Experiment $experiment_id: $wandb_run_name"
        
        # Run single layer experiment with research-based hyperparameters
        python run_graph_classification.py \
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
        
        if [ $? -eq 0 ]; then
            log_message "✅ Experiment $experiment_id completed successfully"
        else
            log_message "❌ Experiment $experiment_id failed"
        fi
        
        experiment_id=$((experiment_id + 1))
    done
done

log_message "🔬 Starting MoE Experiments..."

# MoE Experiments
for layer_combo in "${moe_combinations[@]}"; do
    for dataset in "${datasets[@]}"; do
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
        wandb_run_name="${dataset}_MoE_${clean_combo}_L${num_layer}_H${hidden_dim}_lr${learning_rate}_d${dropout}_exp${experiment_id}"
        
        log_message "🧪 Experiment $experiment_id: $wandb_run_name"
        
        # Run MoE experiment with research-based hyperparameters
        python run_graph_classification.py \
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
        
        if [ $? -eq 0 ]; then
            log_message "✅ Experiment $experiment_id completed successfully"
        else
            log_message "❌ Experiment $experiment_id failed"
        fi
        
        experiment_id=$((experiment_id + 1))
    done
done

log_message "🎉 Comprehensive sweep completed! Total experiments: $((experiment_id - 1))"
