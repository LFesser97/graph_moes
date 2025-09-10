# README:


# TODO / IDEAS:
# Hypergraph encodings > add something to store them too
# Not doing rewiring anymore
# Todo: lint, add mypy and pylint github action



# Graph Mixture of Experts (Graph MoE)

A PyTorch implementation of Graph Neural Networks with Mixture of Experts (MoE) architectures and heterogeneous layer types for graph classification and regression tasks.

## Overview

This repository implements two main research contributions:

1. **Heterogeneneity in Graph Learning**: Investigation of diverse GNN architectures including complex-valued (Unitary) and orthogonal transformations alongside traditional methods
2. **Graph Mixture of Experts (MoE)**: Dynamic routing mechanisms that select between different GNN expert architectures based on graph characteristics

The codebase supports extensive experimentation with different GNN layer types, structural encodings, and curvature-based graph features.

## Key Features

- **Multiple GNN Architectures**: GCN, GIN, SAGE, GAT, MLP, GPS
- **Specialized Layers**: Unitary (complex-valued), Orthogonal transformations
- **Mixture of Experts**: Router networks for dynamic expert selection
- **Structural Encodings**: Curvature-based features, positional encodings
- **Comprehensive Evaluation**: Classification and regression benchmarks

## Repository Structure

```
graph_moes/
├── models/ # Core model implementations
│ ├── graph_moe.py # MoE architectures (MoE, MoE_E)
│ ├── graph_model.py # Classification GNN models
│ ├── graph_regression_model.py # Regression GNN models
│ ├── complex_valued_layers.py # Unitary/complex layers
│ ├── real_valued_layers.py # Orthogonal layers
│ ├── layers.py # Custom layer implementations
│ └── performer.py # Performer attention mechanism
├── experiments/ # Experiment frameworks
│ ├── graph_classification.py # Classification experiment runner
│ └── graph_regression.py # Regression experiment runner
├── custom_encodings.py # Structural encodings (LCP, curvature)
├── GraphRicciCurvature/ # Curvature computation utilities
├── run_graph_classification.py # Main classification script
├── run_graph_regression.py # Main regression script
├── hyperparams.py # Command-line argument parsing
├── measure_smoothing.py # Dirichlet energy computation
├── attention.py # Attention mechanisms
└── results/ # Experimental results and logs
```


## Installation

### Environment Setup


### Option 1: Conda Environment (Recommended)
Use the provided `environment.yml`:

```bash
conda env create -f environment.yml
conda activate borf
```

# this one did not work for me so adding a pyproject.toml


#### Option 2: pip with pyproject.toml
For a modern pip-based installation:

1. Create and activate the conda environment:
```bash
conda create -n moe python=3.11
conda activate moe
```

```
pip install -e .
```

Install these manually:
``` 
# PyTorch Geometric - may need manual installation
# "torch-geometric>=2.3.1",  
# "torch-scatter>=2.0.9",    
# "torch-sparse>=0.6.15",   
# "torch-cluster>=1.6.1",    
```

```
pip install torch-geometric torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

2. Verify PyTorch Geometric installation:
```bash
python -c "import torch_geometric; print(torch_geometric.__version__)"
```


### Manual Installation (Alternative)

If the environment file fails:
```bash
# Core dependencies
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install torch-geometric torch-scatter torch-sparse

# Additional packages
pip install attrdict pandas numpy scipy networkx matplotlib
pip install graphriccicurvature numba tqdm
```

## Quick Start

### Graph Classification

**Basic Example - MUTAG dataset:**
```bash
python run_graph_classification.py \
    --dataset mutag \
    --layer_type GCN \
    --num_trials 10 \
    --num_layers 4 \
    --hidden_dim 64
```

**Mixture of Experts:**
```bash
python run_graph_classification.py \
    --dataset enzymes \
    --layer_types '["GCN", "GIN"]' \
    --num_trials 10 \
    --num_layers 6 \
    --hidden_dim 128
```

**With Structural Encoding:**
```bash
python run_graph_classification.py \
    --dataset proteins \
    --layer_type GIN \
    --encoding LCP \
    --num_trials 5
```

### Graph Regression

**ZINC Molecular Property Prediction:**
```bash
python run_graph_regression.py \
    --dataset zinc \
    --layer_type GINE \
    --num_trials 15 \
    --num_layers 16 \
    --hidden_dim 64 \
    --learning_rate 0.001
```

## Supported Datasets

### Classification Datasets
| Dataset | Graphs | Classes | Description |
|---------|---------|---------|-------------|
| **MUTAG** | 188 | 2 | Mutagenic aromatic compounds |
| **ENZYMES** | 600 | 6 | Protein tertiary structures |
| **PROTEINS** | 1,113 | 2 | Protein structures |
| **IMDB-BINARY** | 1,000 | 2 | Movie collaboration networks |
| **COLLAB** | 5,000 | 3 | Scientific collaboration networks |
| **REDDIT-BINARY** | 2,000 | 2 | Reddit thread discussions |


### Regression Datasets
| Dataset | Graphs | Task | Description |
|---------|---------|------|-------------|
| **ZINC** | 12,000 | Regression | Molecular solubility prediction |

# TODO: Raph: to add MNIST, CIFAR, Pattern

## Model Architectures

### Standard GNN Layers
- **GCN**: Graph Convolutional Network - mean aggregation with smoothness bias
- **GIN**: Graph Isomorphism Network - sum aggregation, theoretically most expressive
- **SAGE**: GraphSAGE - scalable to large graphs with sampling
- **GAT**: Graph Attention Network - attention-based neighbor weighting
- **MLP**: Multi-Layer Perceptron - ignores graph structure (baseline)

# TODO: Raph: add GatedGCN

### Specialized Architectures
- **Unitary**: Complex-valued transformations with unitary constraints
- **Orthogonal**: Real-valued orthogonal transformations
- **GPS**: Hybrid MLP + transformer
- **GINE**: Graph Isomorphism Network 

### Mixture of Experts
- **MoE**: Standard mixture with learned routing
- **MoE_E**: Enhanced MoE with feature masking for input diversity

## Key Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--dataset` | Dataset name | None | mutag, enzymes, proteins, imdb, collab, reddit, zinc |
| `--layer_type` | Single GNN type | MoE | GCN, GIN, SAGE, GAT, MLP, Unitary, GPS, MoE |
| `--layer_types` | Expert types for MoE | None | `'["GCN", "GIN"]'`, `'["GIN", "Unitary"]'` |
| `--num_layers` | Network depth | 4 | 2-16 |
| `--hidden_dim` | Hidden dimension | 64 | 32, 64, 128, 256 |
| `--learning_rate` | Learning rate | 0.001 | 0.0001, 0.001, 0.01 |
| `--dropout` | Dropout rate | 0.1 | 0.0-0.5 |
| `--num_trials` | Number of runs | 10 | 1-50 |
| `--encoding` | Structural encoding | None | LAPE, RWPE, LCP, LDP, SUB, EGO |

## Structural Encodings

### Available Encodings
- **LAPE**: Laplacian Eigenvector Positional Encoding
- **RWPE**: Random Walk Positional Encoding  
- **LCP**: Local Curvature Profile (Ollivier-Ricci curvature statistics)
- **LDP**: Local Degree Profile
- **SUB**: Subgraph-based features
- **EGO**: Ego-network features

Would be fun to add the hg-encodings too!

### Curvature-Based Features
The Local Curvature Profile (LCP) encoding computes Ollivier-Ricci curvature statistics for each node:
- Min, max, mean, std, median of neighbor edge curvatures
- Provides structural information about local graph geometry
- Particularly effective for heterogeneous graph structures

## Advanced Usage

### Router Configuration for MoE
```bash
python run_graph_classification.py \
    --dataset proteins \
    --layer_types '["GCN", "Unitary"]' \
    --router_type GNN \
    --router_layer_type GIN \
    --router_depth 4 \
    --router_dropout 0.1
```

### Deep Network with Unitary Layers
```bash
python run_graph_classification.py \
    --dataset enzymes \
    --layer_type Unitary \
    --num_layers 8 \
    --hidden_dim 128 \
    --patience 100
```

### Hyperparameter Sweeps
Use provided bash scripts for systematic exploration:

```bash
# MoE with GCN+GIN experts
bash moe_gcn_gin_sweep.sh

# MoE with Unitary+GIN experts  
bash moe_uni_gin_sweep.sh
```

## Experimental Results

Results are automatically saved to:
- **CSV files**: `results/graph_classification_{model}_{encoding}.csv`
- **Log files**: `results/graph_classification.txt`
- **Model checkpoints**: `results/{layers}_layers/{dataset}_{model}_{encoding}_graph_dict.pickle`

### Performance Comparison (Test Accuracy %)

| Dataset | GCN | GIN | MoE(GCN+GIN) | Unitary | Improvement |
|---------|-----|-----|--------------|---------|-------------|
| MUTAG | 85.2 | 84.1 | **87.1** | 86.3 | +1.9% |
| ENZYMES | 59.8 | 62.3 | **65.7** | 61.2 | +3.4% |
| PROTEINS | 75.1 | 76.8 | **78.2** | 77.5 | +1.4% |

## Implementation Details

### Mixture of Experts Architecture
```python
# Two expert networks with learned routing
experts = [GNN(args_gcn), GNN(args_gin)]
router = Router(input_dim, num_experts=2)

# Dynamic weighting based on graph features
weights = softmax(router(graph_features))
output = sum(w_i * expert_i(graph) for w_i, expert_i in zip(weights, experts))
```

### Unitary Transformations
```python
# Complex-valued layers with unitary constraints
class UnitaryGCNConvLayer(nn.Module):
    def forward(self, data):
        x_complex = data.x.to(torch.complex64)
        # Apply unitary transformation: U†U = I
        x_new = unitary_transform(x_complex, self.weight)
        return x_new
```
