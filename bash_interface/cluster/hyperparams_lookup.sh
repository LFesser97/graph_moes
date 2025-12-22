#!/bin/bash
# ============================================================================
# Hyperparameter Lookup Function
# ============================================================================
# This script provides a function to retrieve optimal hyperparameters for graph
# neural network experiments based on research paper findings.
# Reference: https://arxiv.org/pdf/2502.09263
#
# The function get_hyperparams(dataset, model_type) returns optimal values for:
#   - hidden_dim: Hidden dimension size
#   - num_layers: Number of graph neural network layers
#   - learning_rate: Learning rate for optimizer
#   - dropout: Dropout rate
#   - batch_size: Batch size for training
#   - epochs: Number of training epochs
#   - patience: Early stopping patience
#
# Supported datasets: mutag, enzymes, proteins, mnist, cifar, pattern, cluster,
#                     zinc, molhiv, molpcba, graphbench_socialnetwork, graphbench_co,
#                     graphbench_sat, graphbench_weather
# Supported model types: GCN, GCN+, GIN, GIN+, and others (defaults for MoE)
#
# Usage: source this file, then call get_hyperparams "dataset" "model_type"
#        Access results via exported variables: HYPERPARAM_HIDDEN_DIM, etc.
# ============================================================================

get_hyperparams() {
    local dataset=$1
    local model_type=$2
    
    # Default values
    hidden_dim=64
    num_layers=4
    learning_rate=0.001
    dropout=0.1
    batch_size=32
    epochs=200
    patience=50
    
    case "$dataset" in
        "mutag")
            case "$model_type" in
                "GCN"|"GCN+")
                    hidden_dim=64
                    num_layers=4
                    learning_rate=0.001
                    dropout=0.1
                    ;;
                "GIN"|"GIN+")
                    hidden_dim=64
                    num_layers=4
                    learning_rate=0.001
                    dropout=0.1
                    ;;
                *)
                    # Default for MoE and other models
                    hidden_dim=64
                    num_layers=4
                    learning_rate=0.001
                    dropout=0.1
                    ;;
            esac
            ;;
            
        "enzymes")
            case "$model_type" in
                "GCN"|"GCN+")
                    hidden_dim=64
                    num_layers=4
                    learning_rate=0.001
                    dropout=0.1
                    ;;
                "GIN"|"GIN+")
                    hidden_dim=64
                    num_layers=4
                    learning_rate=0.001
                    dropout=0.1
                    ;;
                *)
                    hidden_dim=64
                    num_layers=4
                    learning_rate=0.001
                    dropout=0.1
                    ;;
            esac
            ;;
            
        "proteins")
            case "$model_type" in
                "GCN"|"GCN+")
                    hidden_dim=64
                    num_layers=4
                    learning_rate=0.001
                    dropout=0.1
                    ;;
                "GIN"|"GIN+")
                    hidden_dim=64
                    num_layers=4
                    learning_rate=0.001
                    dropout=0.1
                    ;;
                *)
                    hidden_dim=64
                    num_layers=4
                    learning_rate=0.001
                    dropout=0.1
                    ;;
            esac
            ;;
            
        "mnist")
            case "$model_type" in
                "GCN"|"GCN+")
                    hidden_dim=60
                    num_layers=6
                    learning_rate=0.0005
                    dropout=0.15
                    batch_size=16
                    epochs=200
                    ;;
                "GIN"|"GIN+")
                    hidden_dim=60
                    num_layers=5
                    learning_rate=0.001
                    dropout=0.1
                    batch_size=16
                    epochs=200
                    ;;
                *)
                    hidden_dim=60
                    num_layers=5
                    learning_rate=0.001
                    dropout=0.1
                    batch_size=16
                    ;;
            esac
            ;;
            
        "cifar")
            case "$model_type" in
                "GCN"|"GCN+")
                    hidden_dim=65
                    num_layers=5
                    learning_rate=0.001
                    dropout=0.05
                    batch_size=16
                    epochs=200
                    ;;
                "GIN"|"GIN+")
                    hidden_dim=60
                    num_layers=5
                    learning_rate=0.001
                    dropout=0.05
                    batch_size=16
                    epochs=200
                    ;;
                *)
                    hidden_dim=60
                    num_layers=5
                    learning_rate=0.001
                    dropout=0.05
                    batch_size=16
                    ;;
            esac
            ;;
            
        "pattern")
            case "$model_type" in
                "GCN"|"GCN+")
                    hidden_dim=90
                    num_layers=12
                    learning_rate=0.001
                    dropout=0.05
                    batch_size=32
                    epochs=200
                    ;;
                "GIN"|"GIN+")
                    hidden_dim=100
                    num_layers=8
                    learning_rate=0.001
                    dropout=0.05
                    batch_size=32
                    epochs=200
                    ;;
                *)
                    hidden_dim=90
                    num_layers=8
                    learning_rate=0.001
                    dropout=0.05
                    batch_size=32
                    ;;
            esac
            ;;
            
        "cluster")
            case "$model_type" in
                "GCN"|"GCN+")
                    hidden_dim=90
                    num_layers=12
                    learning_rate=0.001
                    dropout=0.1
                    batch_size=16
                    epochs=100
                    ;;
                "GIN"|"GIN+")
                    hidden_dim=90
                    num_layers=10
                    learning_rate=0.0005
                    dropout=0.05
                    batch_size=16
                    epochs=100
                    ;;
                *)
                    hidden_dim=90
                    num_layers=10
                    learning_rate=0.001
                    dropout=0.1
                    batch_size=16
                    ;;
            esac
            ;;
            
        "zinc")
            case "$model_type" in
                "GCN"|"GCN+")
                    hidden_dim=64
                    num_layers=12
                    learning_rate=0.001
                    dropout=0.0
                    batch_size=32
                    epochs=2000
                    patience=250
                    ;;
                "GIN"|"GIN+")
                    hidden_dim=80
                    num_layers=12
                    learning_rate=0.001
                    dropout=0.0
                    batch_size=32
                    epochs=2000
                    patience=250
                    ;;
                *)
                    hidden_dim=64
                    num_layers=12
                    learning_rate=0.001
                    dropout=0.0
                    batch_size=32
                    patience=250
                    ;;
            esac
            ;;
            
        "molhiv")
            case "$model_type" in
                "GCN"|"GCN+")
                    hidden_dim=256
                    num_layers=4
                    learning_rate=0.0001
                    dropout=0.1
                    batch_size=32
                    epochs=100
                    ;;
                "GIN"|"GIN+")
                    hidden_dim=256
                    num_layers=3
                    learning_rate=0.0001
                    dropout=0.0
                    batch_size=32
                    epochs=100
                    ;;
                *)
                    hidden_dim=256
                    num_layers=3
                    learning_rate=0.0001
                    dropout=0.1
                    batch_size=32
                    ;;
            esac
            ;;
            
        "molpcba")
            case "$model_type" in
                "GCN"|"GCN+")
                    hidden_dim=512
                    num_layers=10
                    learning_rate=0.0005
                    dropout=0.2
                    batch_size=512
                    epochs=100
                    ;;
                "GIN"|"GIN+")
                    hidden_dim=300
                    num_layers=16
                    learning_rate=0.0005
                    dropout=0.3
                    batch_size=512
                    epochs=100
                    ;;
                *)
                    hidden_dim=400
                    num_layers=12
                    learning_rate=0.0005
                    dropout=0.2
                    batch_size=512
                    ;;
            esac
            ;;
            
        # GraphBench datasets
        "graphbench_socialnetwork")
            case "$model_type" in
                "GCN"|"GCN+"|"GIN"|"GIN+")
                    hidden_dim=64
                    num_layers=4
                    learning_rate=0.001
                    dropout=0.1
                    batch_size=32
                    epochs=200
                    ;;
                *)
                    hidden_dim=64
                    num_layers=4
                    learning_rate=0.001
                    dropout=0.1
                    batch_size=32
                    ;;
            esac
            ;;
            
        "graphbench_co")
            case "$model_type" in
                "GCN"|"GCN+"|"GIN"|"GIN+")
                    hidden_dim=64
                    num_layers=4
                    learning_rate=0.001
                    dropout=0.1
                    batch_size=32
                    epochs=200
                    ;;
                *)
                    hidden_dim=64
                    num_layers=4
                    learning_rate=0.001
                    dropout=0.1
                    batch_size=32
                    ;;
            esac
            ;;
            
        "graphbench_sat")
            case "$model_type" in
                "GCN"|"GCN+"|"GIN"|"GIN+")
                    hidden_dim=64
                    num_layers=4
                    learning_rate=0.001
                    dropout=0.1
                    batch_size=32
                    epochs=200
                    ;;
                *)
                    hidden_dim=64
                    num_layers=4
                    learning_rate=0.001
                    dropout=0.1
                    batch_size=32
                    ;;
            esac
            ;;
            
        "graphbench_weather")
            case "$model_type" in
                "GCN"|"GCN+"|"GIN"|"GIN+")
                    hidden_dim=64
                    num_layers=4
                    learning_rate=0.001
                    dropout=0.1
                    batch_size=32
                    epochs=200
                    ;;
                *)
                    hidden_dim=64
                    num_layers=4
                    learning_rate=0.001
                    dropout=0.1
                    batch_size=32
                    ;;
            esac
            ;;
            
        *)
            # Default values for other datasets (including other GraphBench datasets)
            hidden_dim=64
            num_layers=4
            learning_rate=0.001
            dropout=0.1
            batch_size=32
            ;;
    esac
    
    # Export the variables so they can be used by calling script
    export HYPERPARAM_HIDDEN_DIM=$hidden_dim
    export HYPERPARAM_NUM_LAYERS=$num_layers
    export HYPERPARAM_LEARNING_RATE=$learning_rate
    export HYPERPARAM_DROPOUT=$dropout
    export HYPERPARAM_BATCH_SIZE=$batch_size
    export HYPERPARAM_EPOCHS=$epochs
    export HYPERPARAM_PATIENCE=$patience
}
