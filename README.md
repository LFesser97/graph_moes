[![Ruff](https://github.com/LFesser97/graph_moes/actions/workflows/ruff.yml/badge.svg)](https://github.com/LFesser97/graph_moes/actions/workflows/ruff.yml)
[![pytest](https://github.com/LFesser97/graph_moes/actions/workflows/pytest.yml/badge.svg)](https://github.com/LFesser97/graph_moes/actions/workflows/pytest.yml)

# TODO (12/10)

* Investigate tmd

top priorities:
* try to increase accuracies to match "Can ... be strong baseline for"
[ with Skip connections or potentially encodings]


add hencodings
Read GPS paper and instead of 50% 50% weight, let an Router decide.
add peptides

1 - add skip connection to eg (GCN, GIN, etc etc)
[It's just adding input to conv layer and adding it to the output]+ 
Check if Dropout is included +
add encodings
check the Can ... be strong baseline for
what we should expect. They also use ENCODINGS!
Look at the paper (main text table 5)

2 - Also look at normalization [Section 3 in the paper].

4 - Compute the hg encodings once and make sure they are fully correct and then upload them.

add to wandb: the router type

2 - Add global attention as an expert.

3 - Try to add Global attention layer as router (imagine MOE as 
extension of GPS)


Later: check that sage does not do better than
all moe connections. 

4 - encodings + hg encodings

(no tmd, no rewiring...)



* Ask Lukas about the weird plots. WHy do we get a very clear separation by indexing?
[ IN PROGRESS]
* Can GNN be strong baselines:
- take these hyperparameters and these datasets. [TODO: check manually]

* Add test suite for accuracies / pytest for all that
* Merge current PR
* Run a new PR with black/isort/ruff





- Redo plotting + TMD
- Adding a few datasets + architectures 
- MoE: foresee how well this is working - look at training curves and that it trains well and accuracy in comparaison with other baselines.
Aiming end of Jan 


# Dataset Status Summary
What dataset we have right now? 
What ran:
proteins, enzymes, imdb, mutag, pattern, reddit, collab, cifar, mnist
TODO: reread what dataset I expect.
Missing:
enzymes, peptides-struct, peptides-func
- total ~20 
MAKE SURE I CAN RERUN ON THE CLUSTER. CLEAN UP.
**Target Datasets (15 total):**
- ✅ **Available & Loaded (14):** ZINC, MNIST, CIFAR10, PATTERN, CLUSTER, Peptides-func, PascalVOC-SP, COCO-SP, MalNet-Tiny, ogbg-molhiv, ogbg-molpcba, ogbg-ppa, ogbg-code2
- ⚠️ **Commented Out (1):** Peptides-struct (needs uncommenting in `run_graph_regression.py`)

**Note:** ogbg-code2 is in classification script but has `output_dim=1` (regression) - verify task type.


IN PROGRESS: TO CODE REVIEW AND CHECK
Vizualisations:
for each dataset and each MOE configurations:
+ code lost for the histograms/TmD
do enough experiments so that each graph is in 10 test datasets.


Maybe idea for later:
- encodings + hg encodings


# END OF TODO (12/10)


# README

# Graph Mixture of Experts (Graph MoE)

A PyTorch implementation of Graph Neural Networks with Mixture of Experts (MoE) architectures and heterogeneous layer types for graph classification and regression tasks.

## TODO

- run locally again and run mamba env export > environment_SAVED.yml
- Get parameters from the paper [DONE]
- Run on the cluster [ABLE!]
 
- TODO / IDEAS:
- Hypergraph encodings > add something to store them too
- Not doing rewiring anymore


Immediate TODOs:
- Reorganise the repo into src/ and scripts/ + split the scripts into smaller files?
for example src/models/moe/routers, src/models/layers, src/models/architectures
src/encodings/curtom_encodings etc. This will help make the repo much more organised and easy to use.
- Add plotting functionality: super important. See the losses (or extract them from wandb),
see the performance of each model of each graph (repeated ten times)
- Fix the missing dataset (ogb causing trouble becuase of sklearn > scipy)
- Todo: add pylint/mypy github action
- Todo: lower priority: add tests

**CURRENT STATUS (December 28, 2025):**

**✅ Currently running in main sweep (8 datasets - Graph Classification)**: enzymes, proteins, mutag, imdb, collab, reddit, mnist, cifar
**📊 Available on WandB (MOE_new project)**: All 8 main sweep datasets above
**❌ Excluded from graph classification**: pattern (node classification - requires different experimental setup), cluster (disabled LRGB dataset)

**🔄 IN PROGRESS - Additional datasets setup:**
- **Script created**: `bash_interface/cluster/comprehensive_sweep_parallel_additional_data.sh` for additional datasets
- **Download script**: `download_ogbg_ppa.py` (modified to auto-approve large downloads)
- **Upload script**: `upload_ogbg_ppa_to_cluster.sh` (for transferring downloaded data to cluster)

**📋 Additional datasets (ready in additional sweep script):**
- **LRGB datasets**: pascalvoc, coco, peptides_func ✅
- **OGB datasets**: ogbg-ppa (once downloaded) ✅
- **GraphBench datasets**: socialnetwork, co, sat, electronic_circuits, chipdesign ✅
- **Excluded**: algorithmic_reasoning_* (too complex), weather (regression), PATTERN (node classification)

**🎯 Next steps:**
1. Download ogbg-ppa dataset on cluster
2. Run additional sweep: `sbatch bash_interface/cluster/comprehensive_sweep_parallel_additional_data.sh`

## 📊 Dataset Types Explanation

**Graph Classification** (Current experiments):
- Each "sample" is an entire graph
- Task: Predict 1 label for the whole graph
- Examples: Enzymes (toxic/non-toxic), MNIST (digit classification), CIFAR (object classification)

**Node Classification** (Separate experiments needed):
- Each "sample" is individual nodes within graphs
- Task: Predict labels for each node in the graph
- Examples: PATTERN (synthetic node patterns), CLUSTER (node clustering), PascalVOC-SP (semantic segmentation)

**Note**: PATTERN dataset requires node-level prediction tasks, not graph-level. A separate experimental setup is available for node classification experiments: `bash_interface/cluster/node_classification_pattern.sh`

**⚠️ Important Distinction:**
- **Graph Classification**: 1 label per graph (our current experiments)
- **Node Classification**: 1 label per node within graphs (PATTERN requires this)

The node classification script handles the different data loading and evaluation required for node-level tasks.

**📝 CURRENT STATUS & NEXT STEPS:**

**✅ ogbg-ppa download**: Script created and ready!

**To download ogbg-ppa on cluster:**
1. Pull latest changes: `git pull`
2. Submit the download job:
   ```bash
   sbatch download_ppa.sh
   ```
3. Monitor the job: `squeue -u rpellegrinext`
4. Check logs: `cat slurm-*.out` (after job completes)
5. The dataset will download directly to cluster storage

**Alternative manual approach:**
```bash
export PYTHONPATH="$(pwd):$(pwd)/src:${PYTHONPATH}"
python scripts/run_graph_classification.py --dataset ppa --layer_type GCN --num_trials 1
```
(Answer "y" to both prompts when asked)

**After ogbg-ppa is available:**
1. Uncomment LRGB datasets (pascalvoc, coco, peptides_func) in `run_graph_classification.py`
2. Submit `comprehensive_sweep_parallel_additional_data.sh` on cluster
3. Test GraphBench datasets individually before adding to sweep


## Overview

This repository implements two main research contributions:

1. **Heterogeneneity in Graph Learning**: Investigation of diverse GNN architectures including complex-valued (Unitary) and orthogonal transformations alongside traditional methods. Exploring which graph are always misclassified.
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


#### Option 1: Conda Environment (Recommended)
Use the provided `environment.yml`:

```bash
conda env create -f environment.yml
conda activate borf
```

this one did not work for me so adding a pyproject.toml:


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


### On the cluser:

```
# 1. Set up environment variables for lab space
export CONDA_PKGS_DIRS=/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/conda/pkgs
export CONDA_ENVS_PATH=/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/conda/envs

# 2. Create the directories
mkdir -p $CONDA_PKGS_DIRS $CONDA_ENVS_PATH

# 3. Load Python module (this includes mamba)
module load python/3.10.12-fasrc01

# 4. Create moe environment
mamba create -n moe python=3.10 pip wheel -y

# 5. Activate the environment
source activate moe

# 6. Install packages with mamba (faster for most packages)
mamba install -y numpy pandas tqdm

# 7. Install PyTorch and PyTorch Geometric
mamba install -y pytorch pytorch-geometric -c pytorch -c pyg

# 8. Install packages only available via pip
pip install wandb attrdict

# 9. Test everything works
python -c "import numpy, torch, torch_geometric, wandb, attrdict; print('✅ moe environment created successfully!')"

# 10. Check new environment
conda info --envs
```

2. Verify PyTorch Geometric installation:
```bash
python -c "import torch_geometric; print(torch_geometric.__version__)"
```

ON THE CLUSER I HAD scikit-learn==1.2.2 and this was working!


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
python scripts/run_graph_classification.py \
    --dataset mutag \
    --layer_type GCN \
    --num_trials 10 \
    --num_layers 4 \
    --hidden_dim 64
```

**Mixture of Experts:**
```bash
python scripts/run_graph_classification.py \
    --dataset enzymes \
    --layer_types '["GCN", "GIN"]' \
    --num_trials 10 \
    --num_layers 6 \
    --hidden_dim 128
```

**With Structural Encoding:**
```bash
python scripts/run_graph_classification.py \
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

### Overview of the datasets used for graph-level tasks.

| Dataset        | # graphs | Avg. # nodes | Avg. # edges | Task Type             |
|----------------|----------|--------------|--------------|-----------------------|
| ZINC           | 12,000   | 23.2         | 24.9         | Graph regression      |
| MNIST          | 70,000   | 70.6         | 564.5        | Graph classification  |
| CIFAR10        | 60,000   | 117.6        | 941.1        | Graph classification  |
| PATTERN        | 14,000   | 118.9        | 3,039.3      | Inductive node cls.   |
| CLUSTER        | 12,000   | 117.2        | 2,150.9      | Inductive node cls.   |
| Peptides-func  | 15,535   | 150.9        | 307.3        | Graph classification  |
| Peptides-struct| 15,535   | 150.9        | 307.3        | Graph regression      |
| PascalVOC-SP   | 11,355   | 479.4        | 2,710.5      | Inductive node cls.   |
| COCO-SP        | 123,286  | 476.9        | 2,693.7      | Inductive node cls.   |
| MalNet-Tiny    | 5,000    | 1,410.3      | 2,859.9      | Graph classification  |
| ogbg-molhiv    | 41,127   | 25.5         | 27.5         | Graph classification  |
| ogbg-molpcba   | 437,929  | 26.0         | 28.1         | Graph classification  |
| ogbg-ppa       | 158,100  | 243.4        | 2,266.1      | Graph classification  |
| ogbg-code2     | 452,741  | 125.2        | 124.2        | Graph classification  |


### Classification Datasets

#### TU Datasets (Traditional Graph Classification)
| Dataset           | Graphs | Classes | Description                       |
|-------------------|--------|---------|-----------------------------------|
| **MUTAG**         | 188    | 2       | Mutagenic aromatic compounds      |
| **ENZYMES**       | 600    | 6       | Protein tertiary structures       |
| **PROTEINS**      | 1,113  | 2       | Protein structures                |
| **IMDB-BINARY**   | 1,000  | 2       | Movie collaboration networks      |
| **COLLAB**        | 5,000  | 3       | Scientific collaboration networks |
| **REDDIT-BINARY** | 2,000  | 2       | Reddit thread discussions         |

#### GNN Benchmark Datasets (Computer Vision)
| Dataset       | Graphs  | Classes | Description                             |
|---------------|---------|---------|-----------------------------------------|
| **MNIST**     | 70,000  | 10      | Handwritten digits as superpixel graphs |
| **CIFAR10**   | 60,000  | 10      | Natural images as superpixel graphs     |
| **PATTERN**   | 14,000  | 2       | Synthetic node classification patterns  |

#### Long Range Graph Benchmark (LRGB) - Available
| Dataset                | Graphs  | Classes | Description                                   |
|------------------------|---------|---------|-----------------------------------------------|
| **PeptidesFunctional** | ~15,000 | 10      | Peptide functional prediction (commented out) |
| **CLUSTER**            | 12,000  | 2       | Inductive node classification                 |

#### Open Graph Benchmark (OGB) - Available
| Dataset          | Graphs  | Classes | Description                      |
|------------------|---------|---------|----------------------------------|
| **ogbg-molhiv**  | 41,127  | 2       | HIV inhibition prediction        |
| **ogbg-molpcba** | 437,929 | 128     | Molecular bioactivity prediction |

#### GraphBench Datasets - Successfully Downloaded
| Dataset                      | Status | Description                                   |
|------------------------------|--------|-----------------------------------------------|
| **sat**                      | ✅     | SAT solving problems                          |
| **chipdesign**               | ✅     | Chip design optimization                      |
| **algorithmic_reasoning_easy** | ✅  | Algorithmic reasoning (easy difficulty)       |

Note: Other GraphBench datasets (socialnetwork, co, algorithmic_reasoning_medium/hard, electronic_circuits) had download issues and may require retry.

### Regression Datasets
| Dataset                | Graphs  | Task       | Description                                   |
|------------------------|---------|------------|-----------------------------------------------|
| **ZINC**               | 12,000  | Regression | Molecular solubility prediction               |
| **PeptidesStructural** | ~15,000 | Regression | Peptide structural prediction (commented out) |

## Model Architectures

### Standard GNN Layers
- **GCN**: Graph Convolutional Network - mean aggregation with smoothness bias
- **GIN**: Graph Isomorphism Network - sum aggregation, theoretically most expressive
- **SAGE**: GraphSAGE - scalable to large graphs with sampling
- **GAT**: Graph Attention Network - attention-based neighbor weighting
- **MLP**: Multi-Layer Perceptron - ignores graph structure (baseline)


### Specialized Architectures
- **Unitary**: Complex-valued transformations with unitary constraints
- **Orthogonal**: Real-valued orthogonal transformations
- **GPS**: Hybrid MLP + transformer
- **GINE**: Graph Isomorphism Network 

### Mixture of Experts
- **MoE**: Standard mixture with learned routing
- **MoE_E**: Enhanced MoE with feature masking for input diversity

## Key Parameters

| Parameter         | Description          | Default | Options |
|-------------------|----------------------|---------|---------|
| `--dataset`       | Dataset name         | None    | mutag, enzymes, proteins, imdb, collab, reddit, zinc |
| `--layer_type`    | Single GNN type      | MoE     | GCN, GIN, SAGE, GAT, MLP, Unitary, GPS, MoE |
| `--layer_types`   | Expert types for MoE | None    | `'["GCN", "GIN"]'`, `'["GIN", "Unitary"]'` |
| `--num_layers`    | Network depth        | 4       | 2-16 |
| `--hidden_dim`    | Hidden dimension     | 64      | 32, 64, 128, 256 |
| `--learning_rate` | Learning rate        | 0.001   | 0.0001, 0.001, 0.01 |
| `--dropout`       | Dropout rate         | 0.1     | 0.0-0.5 |
| `--num_trials`    | Number of runs       | 10      | 1-50 |
| `--encoding`      | Structural encoding  | None    | LAPE, RWPE, LCP, LDP, SUB, EGO |

## Structural Encodings

### Available Encodings
- **LAPE**: Laplacian Eigenvector Positional Encoding
- **RWPE**: Random Walk Positional Encoding  
- **LCP**: Local Curvature Profile (Ollivier-Ricci curvature statistics)
- **LDP**: Local Degree Profile
- **SUB**: Subgraph-based features
- **EGO**: Ego-network features

Would be fun to add the hypergraph-encodings too!

### Curvature-Based Features
The Local Curvature Profile (LCP) encoding computes Ollivier-Ricci curvature statistics for each node:
- Min, max, mean, std, median of neighbor edge curvatures
- Provides structural information about local graph geometry
- Particularly effective for heterogeneous graph structures

## Advanced Usage

### Router Configuration for MoE
```bash
python scripts/run_graph_classification.py \
    --dataset proteins \
    --layer_types '["GCN", "Unitary"]' \
    --router_type GNN \
    --router_layer_type GIN \
    --router_depth 4 \
    --router_dropout 0.1
```

### Deep Network with Unitary Layers
```bash
python scripts/run_graph_classification.py \
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
bash bash_interface/local/moe_gcn_gin_sweep.sh

# MoE with Unitary+GIN experts  
bash bash_interface/local/moe_uni_gin_sweep.sh
```

(there are sister scripts for running on the cluster).

## Experimental Results

Results are automatically saved to:
- **CSV files**: `results/graph_classification_{model}_{encoding}.csv`
- **Log files**: `results/graph_classification.txt`
- **Model checkpoints**: `results/{layers}_layers/{dataset}_{model}_{encoding}_graph_dict.pickle`

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


