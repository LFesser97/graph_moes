# Tree Mover's Distance (TMD) Implementation Plan

## Overview

The **Tree Mover's Distance (TMD)** is a similarity measure on graphs that jointly evaluates feature and topological information (Chuang & Jegelka, 2022). Like Message-Passing Graph Neural Networks (MPGNNs), it views a graph as a set of computation trees. TMD compares graphs by characterizing the similarity of their computation trees via hierarchical optimal transport.

## Theoretical Background

### Computation Trees

A **computation tree** for a node `v` in a graph `G` is constructed by:
1. Starting with node `v` as the root
2. Adding the node's neighbors to the first level
3. Recursively adding neighbors of nodes at each level to the next level
4. This creates a hierarchical tree structure rooted at `v`

Each node in the graph has its own computation tree, and the graph can be viewed as a collection of these trees.

### Tree Mover's Distance

The TMD between two graphs `G₁` and `G₂` is computed by:

1. **Tree Construction**: For each graph, construct computation trees for all nodes
2. **Tree Comparison**: Compare pairs of computation trees using hierarchical optimal transport:
   - Compare the root nodes (using node features)
   - Recursively compare subtrees
   - Compute the optimal transport cost between trees
3. **Graph Comparison**: Aggregate the tree-level distances to compute the overall graph distance

The similarity of two trees `T_v` and `T_u` is computed by:
- Comparing their roots `f_v` and `f_u` (node features)
- Recursively comparing their subtrees
- Using optimal transport to find the minimum cost alignment

### Key Properties

- **Symmetric**: TMD(G₁, G₂) = TMD(G₂, G₁)
- **Non-negative**: TMD(G₁, G₂) ≥ 0
- **Feature-aware**: Incorporates node features, not just structure
- **Hierarchical**: Captures multi-scale structural information

## Class-Distance Ratio

### Definition

Given a graph dataset `D = {G₁, ..., Gₙ}` and labels `Y = {Y₁, ..., Yₙ}`, the **class-distance ratio** of a graph `Gᵢ` is defined as:

```
ρ(Gᵢ) = min{TMD(Gᵢ, Gⱼ) : Yᵢ = Yⱼ, Gⱼ ∈ D\{Gᵢ}}
        ──────────────────────────────────────────────
        min{TMD(Gᵢ, Gⱼ) : Yᵢ ≠ Yⱼ, Gⱼ ∈ D}
```

In words:
- **Numerator**: Minimum TMD to a graph with the **same label** as `Gᵢ`
- **Denominator**: Minimum TMD to a graph with a **different label** than `Gᵢ`

### Interpretation

- **ρ(Gᵢ) < 1**: The graph is closer to graphs of the same class than to graphs of different classes → **Good class separation**
- **ρ(Gᵢ) = 1**: The graph is equidistant to same-class and different-class graphs → **Ambiguous classification**
- **ρ(Gᵢ) > 1**: The graph is closer to graphs of different classes → **Poor class separation** (potential mislabeling or hard example)

### Use Cases

- **Dataset Analysis**: Understand the difficulty of classification tasks
- **Outlier Detection**: Identify graphs with high class-distance ratios
- **Data Quality**: Assess label quality and class separability
- **Model Evaluation**: Compare TMD-based separability with model performance

### Connection to GNN Performance

Empirical studies have shown a **highly significant negative correlation** between a graph's class-distance ratio and its average GNN performance:

- **Mutag dataset**: Pearson correlation coefficient of **-0.441**
- **Enzymes dataset**: Pearson correlation coefficient of **-0.124**

This means that **graphs with higher class-distance ratios have much lower average accuracies**, i.e., they are much harder to classify correctly.

#### Theoretical Foundation: Theorem 3.1

The following theorem (Chuang & Jegelka, 2022, Theorem 8) provides a theoretical explanation for this observation:

**Theorem 3.1**: Given an L-layer GNN `h : X → R` and two graphs `G, G′ ∈ D`, we have:

```
∥h(G) - h(G′)∥ ≤ ∏_{l=1}^{L+1} K^{(l)}_φ · TMD_{L+1}^w(G, G′)
```

where `w^{(l)} = ε · P_{l-1}^{L+1} / P_l^{L+1}` for all `l ≤ L` and `P_l^L` is the l-th number at level L of Pascal's triangle.

**Interpretation**:
- This result indicates that a GNN's prediction on a graph `G` **cannot diverge too far** from its prediction on a similar (in the sense of TMD) graph `G′`.
- A value of **ρ(G) > 1** therefore indicates that the GNN prediction cannot be too far from a prediction made on a graph with a **different label**.
- Consequently, **graphs with ρ(G) > 1 are hard to classify correctly** because the GNN's output is constrained to be similar to graphs from different classes.

**Practical Implications**:
- Class-distance ratio can serve as a **predictor of classification difficulty** before training
- Graphs with high ρ values may require special attention (e.g., data augmentation, different architectures)
- The correlation analysis can help identify which graphs are inherently difficult for GNNs

## Official Implementation

An official implementation of TMD is available at: **[https://github.com/chingyaoc/TMD](https://github.com/chingyaoc/TMD)**

### Usage Example

```python
from tmd import TMD
from torch_geometric.datasets import TUDataset

dataset = TUDataset('data', name='MUTAG')
d = TMD(dataset[0], dataset[1], w=1.0, L=4)
```

**Parameters:**
- `w`: Weighting constant (can be a single float or a list of weights for each layer)
- `L`: Maximum depth of computation trees (number of layers)

**Weighted TMD** (for tighter bounds per Theorem 8):
```python
d = TMD(dataset[0], dataset[1], w=[0.33, 1, 3], L=4)
```
Note: `len(w)` must equal `L-1`.

### Implementation Strategy

We have two options:
1. **Use the official implementation directly** (recommended for initial implementation)
   - Clone or include their `tmd.py` module
   - Adapt it to our codebase structure
   - Ensure compatibility with our PyTorch Geometric Data objects

2. **Implement our own version** based on their approach
   - Study their implementation
   - Re-implement with our coding standards (type hints, docstrings, mypy compliance)
   - Integrate with our existing utilities

**Recommendation**: Start with option 1 (use/adapt their code) for faster implementation, then consider option 2 if we need customizations.

## Implementation Plan

### Phase 1: Core TMD Implementation

#### 1.1 Module Structure
Create a new module: `src/graph_moes/utils/tree_mover_distance.py`

**Approach**: Adapt the official implementation from [chingyaoc/TMD](https://github.com/chingyaoc/TMD) to our codebase.

**Key Components:**
- `TMD()`: Main function to compute TMD between two graphs (from official implementation)
- `compute_tmd_matrix()`: Compute pairwise TMD for a dataset
- `compute_class_distance_ratios()`: Compute class-distance ratios for all graphs
- Helper functions for integration with our dataset loading

#### 1.2 Integration Steps
1. **Obtain Official Implementation**:
   - Download `tmd.py` from [https://github.com/chingyaoc/TMD](https://github.com/chingyaoc/TMD)
   - Review the implementation to understand the algorithm

2. **Adapt to Our Codebase**:
   - Ensure compatibility with PyTorch Geometric `Data` objects (should already work)
   - Add type hints and docstrings for mypy compliance
   - Follow our coding conventions (see user rules)
   - Test with our dataset loading infrastructure

3. **Dependencies**:
   - ✅ `pot>=0.9.0` is already in `pyproject.toml` (line 68)
   - ✅ PyTorch Geometric is already installed
   - No additional dependencies needed!

#### 1.3 Understanding the Official Implementation
The official `TMD()` function:
- Takes two PyTorch Geometric `Data` objects
- Constructs computation trees for all nodes (up to depth `L`)
- Uses hierarchical optimal transport via the `POT` library
- Returns the TMD distance as a scalar

**Key Implementation Details** (from official code):
- Uses `ot.emd2()` (Earth Mover's Distance) from POT library for optimal transport
- Recursively compares computation trees level by level
- Handles node features and graph structure jointly

### Phase 2: Class-Distance Ratio Computation

#### 2.1 Dataset-Level Analysis
Create: `scripts/compute_tmd_analysis.py`

**Functionality**:
- Load a dataset (using existing dataset loading infrastructure from `scripts/run_graph_classification.py`)
- **Pairwise Distance Computation** (can use official `pairwise_dist.py` as reference):
  - Compute TMD for all graph pairs (can be parallelized)
  - Save distance matrix for reuse
  - Handle large datasets with batching
- For each graph `Gᵢ`:
  - Find minimum TMD to same-class graphs
  - Find minimum TMD to different-class graphs
  - Compute class-distance ratio `ρ(Gᵢ)`
- Aggregate statistics:
  - Mean, median, std of ratios
  - Distribution of ratios
  - Graphs with ρ > 1 (potential issues)
- **Correlation Analysis** (if GNN performance data available):
  - Load per-graph GNN accuracies (from `track_avg_accuracy` utilities)
  - Compute Pearson correlation between class-distance ratios and average accuracies
  - Generate scatter plots: ρ(Gᵢ) vs. average accuracy
  - Identify hard examples (high ρ, low accuracy)

#### 2.2 Efficient Computation
- **Pairwise Distance Matrix**: Pre-compute TMD for all graph pairs
- **Parallelization**: Use multiprocessing for independent TMD computations
- **Caching**: Save computed TMD values to avoid recomputation
- **Sampling**: For very large datasets, consider sampling strategies

### Phase 3: Integration and Utilities

#### 3.1 Integration Points
- **Dataset Loading**: Use existing dataset loading from `scripts/run_graph_classification.py`
- **Graph Format**: Work with PyTorch Geometric `Data` objects
- **Results Storage**: Save results in `results/` directory

#### 3.2 Utility Functions
- `compute_tmd_matrix()`: Compute full pairwise TMD matrix for a dataset
- `compute_class_distance_ratios()`: Compute ratios for all graphs
- `analyze_tmd_results()`: Generate statistics and visualizations
- `save_tmd_results()`: Save results to CSV/JSON
- `plot_class_distance_ratios_violin()`: Create violin plot of class-distance ratios
  ```python
  def plot_class_distance_ratios_violin(
      ratios: np.ndarray,
      dataset_name: str,
      labels: Optional[np.ndarray] = None,  # Optional: for grouped violin plot
      output_dir: str = "visualizations/tmd",
  ) -> str:
      """Plot violin plot of class-distance ratios.
      
      Args:
          ratios: Array of class-distance ratios ρ(Gᵢ) for all graphs
          dataset_name: Name of the dataset
          labels: Optional array of class labels for grouped visualization
          output_dir: Directory to save the plot
      
      Returns:
          Path to saved plot file
      """
  ```

#### 3.3 Visualization
- **Violin plot of class-distance ratios**: Show distribution of ρ(Gᵢ) values across the dataset
  - **Single violin plot**: Overall distribution of all class-distance ratios
  - **Grouped violin plot** (optional): Group by class label to see if certain classes have higher ratios
  - Shows median, quartiles, and distribution shape
  - Highlight threshold at ρ = 1 (graphs with ρ > 1 are harder to classify)
  - Save to `visualizations/tmd/{dataset}_class_distance_ratios_violin.png`
- Histogram of class-distance ratios
- Scatter plot: TMD to same class vs. TMD to different class
- Identify and visualize outliers (high ρ values)
- **Correlation visualization**: Scatter plot of class-distance ratio vs. average GNN accuracy
- **Hard examples identification**: Highlight graphs with high ρ and low accuracy

**Visualization Implementation Notes**:
- Use `matplotlib.pyplot.violinplot()` or `seaborn.violinplot()` (if seaborn is added)
- Include summary statistics (mean, median, std) in plot title or text box
- Use color coding: green for ρ < 1 (easy), yellow for ρ ≈ 1 (ambiguous), red for ρ > 1 (hard)

### Phase 4: Testing and Validation

#### 4.1 Unit Tests
- Test computation tree construction
- Test tree-to-tree distance computation
- Test TMD on simple known graphs
- Test class-distance ratio computation

#### 4.2 Validation
- Compare with known graph distances (if available)
- Verify symmetry: TMD(G₁, G₂) = TMD(G₂, G₁)
- Check that TMD(G, G) = 0 (or very small)
- Validate on small datasets first

## File Structure

```
src/graph_moes/utils/
  └── tree_mover_distance.py    # Core TMD implementation

scripts/
  └── compute_tmd_analysis.py   # Dataset-level TMD analysis

tests/
  └── test_tree_mover_distance.py  # Unit tests for TMD

results/
  └── tmd_analysis/              # TMD computation results
      ├── {dataset}_tmd_matrix.npy
      ├── {dataset}_class_ratios.csv
      └── {dataset}_tmd_stats.json

visualizations/
  └── tmd/                        # TMD visualizations
      ├── {dataset}_class_distance_ratios_violin.png
      ├── {dataset}_class_distance_ratios_histogram.png
      ├── {dataset}_tmd_correlation_scatter.png
      └── {dataset}_tmd_same_vs_diff_scatter.png
```

## Dependencies

**Dependencies Status**:
- ✅ `pot>=0.9.0`: Already in `pyproject.toml` (line 68) - Python Optimal Transport library
- ✅ `scipy>=1.9.1`: Already in `pyproject.toml` (line 30)
- ✅ `numpy>=1.23.1,<2.0`: Already in `pyproject.toml` (line 27)
- ✅ `torch-geometric>=2.3.1`: Already in `pyproject.toml` (line 44)

**No new dependencies needed!** The codebase already has all required libraries.

## Implementation Steps

1. **Step 1**: Obtain and integrate official TMD implementation
   - Download `tmd.py` from [https://github.com/chingyaoc/TMD](https://github.com/chingyaoc/TMD)
   - Place in `src/graph_moes/utils/tree_mover_distance.py`
   - Add type hints and docstrings for mypy compliance
   - Test basic functionality: `TMD(graph1, graph2, w=1.0, L=4)`

2. **Step 2**: Create wrapper functions for our use case
   - `compute_tmd_matrix()`: Compute pairwise TMD for a dataset
   - `compute_class_distance_ratios()`: Compute ratios for all graphs
   - Integrate with our dataset loading infrastructure

3. **Step 3**: Optimize for larger datasets
   - Add parallelization (reference `pairwise_dist.py` from official repo)
   - Add caching of distance matrices
   - Handle batching for very large datasets

4. **Step 4**: Implement class-distance ratio computation
   - Use pre-computed TMD matrix
   - Compute ratios for all graphs
   - Generate statistics and analysis

5. **Step 5**: Add correlation analysis with GNN performance
   - Load per-graph accuracies from `track_avg_accuracy` utilities
   - Compute Pearson correlation
   - Generate visualizations

6. **Step 6**: Create analysis script
   - `scripts/compute_tmd_analysis.py`
   - Dataset loading integration
   - Batch processing
   - Results saving (CSV, JSON, plots)
   - **Violin plot generation**: Create `plot_class_distance_ratios()` function
     - Input: array of class-distance ratios, optional class labels for grouping
     - Output: violin plot saved to `visualizations/tmd/` directory
     - Show distribution, median, quartiles, and outliers

7. **Step 7**: Add tests and validation
   - Unit tests for TMD computation
   - Test symmetry: TMD(G₁, G₂) = TMD(G₂, G₁)
   - Test identity: TMD(G, G) ≈ 0
   - Validate on Mutag and Enzymes datasets

8. **Step 8**: Documentation and examples
   - Code documentation
   - Usage examples in README or separate guide
   - Results interpretation guide

## References

- Chuang, C. Y., & Jegelka, S. (2022). Tree Mover's Distance: Bridging Graph Metrics and Stability of Graph Neural Networks. *Advances in Neural Information Processing Systems*, 35.

- **Official Implementation**: [https://github.com/chingyaoc/TMD](https://github.com/chingyaoc/TMD)
  - Main TMD computation: `tmd.py`
  - Pairwise distance computation: `pairwise_dist.py`
  - Stability experiments: `stability.py`
  - SVM classification: `svc.py`

## Notes

- **Computational Complexity**: TMD computation can be expensive for large graphs. Consider:
  - Limiting tree depth (e.g., 3-5 levels)
  - Sampling nodes for tree construction
  - Using approximate optimal transport methods
  - Parallelizing independent computations

- **Node Features**: TMD requires node features. For graphs without features, consider:
  - Using degree as a feature
  - Using one-hot encoding of node types
  - Using learned embeddings

- **Edge Features**: Current plan focuses on node features. Edge features can be incorporated in tree construction if needed.

