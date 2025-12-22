#!/usr/bin/env bash
# ============================================================================
# MoE GIN+Unitary Hyperparameter Sweep - Local Execution
# ============================================================================
# This script runs a hyperparameter sweep for MoE (Mixture of Experts) experiments
# combining GIN and Unitary layer types. Designed for local execution (not cluster).
# It sequentially tests different hyperparameter combinations.
#
# The script sweeps over:
#   - Datasets: proteins, mutag (2 datasets)
#   - Learning rates: 0.001, 0.0001 (2 values)
#   - Hidden dimensions: 64, 128 (2 values)
#   - Number of layers: 4, 6 (2 values)
#   - Dropout rates: 0.0, 0.1, 0.2 (3 values)
#
# Total combinations per dataset: 2 × 2 × 2 × 3 = 24 experiments
# Total experiments: 2 datasets × 24 = 48 experiments
# Each experiment runs 5 trials.
#
# All results are logged to WandB with tags: ["local", "sweep", "gin_unitary"]
#
# Usage: ./moe_uni_gin_sweep.sh
# ============================================================================

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

# List of datasets in table
# conv_types=(gcn dir-gcn uni dir-uni)
datasets=(proteins mutag)

# # All available datasets from scripts/run_graph_classification.py
# datasets=(mutag enzymes proteins imdb collab reddit mnist cifar pattern cluster pascalvoc coco molhiv molpcba)


for dataset in "${datasets[@]}"; do
    learning_rates=(0.001 0.0001)
    hidden_dims=(64 128)
    num_layers=(4 6)
    dropouts=(0.0 0.1 0.2)

    # Loop over each hyperparameter combination for the current dataset
    for learning_rate in "${learning_rates[@]}"; do
        for hidden_dim in "${hidden_dims[@]}"; do
            for num_layer in "${num_layers[@]}"; do
                for dropout in "${dropouts[@]}"; do
                    # Generate wandb run name for this specific combination
                    wandb_run_name="${dataset}_GIN_Unitary_L${num_layer}_H${hidden_dim}_lr${learning_rate}_d${dropout}_local"
                    
                    echo "Running MoE with WandB, lr=${learning_rate}, hidden_dim=${hidden_dim}, num_layers=${num_layer}, dropout=${dropout} on ${dataset}"
                    echo "WandB run name: $wandb_run_name"
                    
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
                        --wandb_tags '["local", "sweep", "gin_unitary"]'
                done
            done
        done
    done
done