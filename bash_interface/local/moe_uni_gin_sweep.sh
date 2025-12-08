#!/usr/bin/env bash

# A script to sweep over specific hyperparameters for multiple datasets
# and run: python -m src.run --num_runs 10 --model dir-uni

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