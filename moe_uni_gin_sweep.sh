#!/usr/bin/env bash

# A script to sweep over specific hyperparameters for multiple datasets
# and run: python -m src.run --num_runs 10 --model dir-uni

# List of datasets in your table
# conv_types=(gcn dir-gcn uni dir-uni)
datasets=(proteins mutag)

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
                    echo "Running MoE, lr=${learning_rate}, hidden_dim=${hidden_dim}, num_layers=${num_layer}, dropout=${dropout} on ${dataset}"
                    python run_graph_classification.py \
                        --num_trials 5 \
                        --dataset "$dataset" \
                        --learning_rate "$learning_rate" \
                        --hidden_dim "$hidden_dim" \
                        --num_layers "$num_layer" \
                        --dropout "$dropout" \
                        --patience 50 \
                        --layer_types '["GIN", "Unitary"]'
                done
            done
        done
    done
done