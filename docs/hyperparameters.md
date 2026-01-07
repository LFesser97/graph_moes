
All from:

https://arxiv.org/pdf/2502.09263

"Can Classic GNNs Be Strong Baselines for Graph-level Tasks?
Simple Architectures Meet Excellence"


Those are coded in the bash script:
/Users/pellegrinraphael/Desktop/Academic_Research/Repos_GNN/graph_moes/bash_interface/cluster/hyperparams_lookup.sh


# GCN+

## Table 8. Hyperparameter settings of GCN+ on benchmarks from (Dwivedi et al., 2023).

| Hyperparameter        | ZINC   | MNIST  | CIFAR10 | PATTERN | CLUSTER |
|-----------------------|--------|--------|---------|---------|---------|
| # GNN Layers          | 12     | 6      | 5       | 12      | 12      |
| Edge Feature Module   | True   | True   | True    | True    | False   |
| Normalization         | BN     | BN     | BN      | BN      | BN      |
| Dropout               | 0.0    | 0.15   | 0.05    | 0.05    | 0.1     |
| Residual Connections  | True   | True   | True    | True    | True    |
| FFN                   | True   | True   | True    | True    | True    |
| PE                    | RWSE-32| False  | False   | RWSE-32 | RWSE-20 |
| Hidden Dim            | 64     | 60     | 65      | 90      | 90      |
| Graph Pooling         | add    | mean   | mean    | -      | -       |
| Batch Size            | 32     | 16     | 16      | 32      | 16      |
| Learning Rate         | 0.001  | 0.0005 | 0.001   | 0.001   | 0.001   |
| # Epochs              | 2000   | 200    | 200     | 200     | 100     |
| # Warmup Epochs       | 50     | 5      | 5       | 5       | 5       |
| Weight Decay          | 1e-5   | 1e-5   | 1e-5    | 1e-5    | 1e-5    |
| # Parameters          | 260,177| 112,570| 114,345 | 517,219 | 516,674 |
| Time (epoch)          | 7.6s   | 60.1s  | 40.2s   | 19.5s   | 29.7s   |



## Table 9. Hyperparameter settings of GCN+ on LRGB and OGB datasets.

| Hyperparameter        | Peptides-func | Peptides-struct | PascalVOC-SP | COCO-SP | MalNet-Tiny | ogbg-molhiv | ogbg-molpcba | ogbg-ppa | ogbg-code2 |
|-----------------------|---------------|-----------------|--------------|---------|-------------|-------------|--------------|----------|------------|
| # GNN Layers          | 3             | 5               | 14           | 18      | 8           | 4           | 10           | 4        | 4          |
| Edge Feature Module   | True          | False           | True         | True    | True        | True        | True         | True     | True       |
| Normalization         | BN            | BN              | BN           | BN      | BN          | BN          | BN           | BN       | BN         |
| Dropout               | 0.2           | 0.2             | 0.1          | 0.05    | 0.0         | 0.1         | 0.2          | 0.2      | 0.2        |
| Residual Connections  | False         | False           | True         | True    | True        | False       | False        | True     | True       |
| FFN                   | False         | False           | True         | True    | True        | True        | True         | True     | True       |
| PE                    | RWSE-32       | RWSE-32         | False        | False   | False       | RWSE-20     | RWSE-16      | False    | False      |
| Hidden Dim            | 275           | 255             | 85           | 70      | 110         | 256         | 512          | 512      | 512        |
| Graph Pooling         | mean          | mean            | â            | â       | max         | mean        | mean         | mean     | mean       |
| Batch Size            | 16            | 32              | 50           | 50      | 16          | 32          | 512          | 32       | 32         |
| Learning Rate         | 0.001         | 0.001           | 0.001        | 0.001   | 0.0005      | 0.0001      | 0.0005       | 0.0003   | 0.0001     |
| # Epochs              | 300           | 300             | 200          | 300     | 150         | 100         | 100          | 400      | 30         |
| # Warmup Epochs       | 5             | 5               | 10           | 10      | 10          | 5           | 5            | 10       | 2          |
| Weight Decay          | 0.0           | 0.0             | 0.0          | 0.0     | 1e-5        | 1e-5        | 1e-5         | 1e-5     | 1e-6       |
| # Parameters          | 507,351       | 506,127         | 520,986      | 460,611 | 494,235     | 1,407,641   | 13,316,700   | 5,549,605| 23,291,826 |
| Time (epoch)          | 6.9s          | 6.6s            | 12.5s        | 162.5s  | 6.6s        | 16.3s       | 91.4s        | 178.2s   | 476.3s     |

# GIN+

## Table 10. Hyperparameter settings of GIN+ on benchmarks from (Dwivedi et al., 2023).

| Hyperparameter        | ZINC   | MNIST  | CIFAR10 | PATTERN | CLUSTER |
|-----------------------|--------|--------|---------|---------|---------|
| # GNN Layers          | 12     | 5      | 5       | 8       | 10      |
| Edge Feature Module   | True   | True   | True    | True    | True    |
| Normalization         | BN     | BN     | BN      | BN      | BN      |
| Dropout               | 0.0    | 0.1    | 0.05    | 0.05    | 0.05    |
| Residual Connections  | True   | True   | True    | True    | True    |
| FFN                   | True   | True   | True    | True    | True    |
| PE                    | RWSE-20| False  | False   | RWSE-32 | RWSE-20 |
| Hidden Dim            | 80     | 60     | 60      | 100     | 90      |
| Graph Pooling         | sum    | mean   | mean    | -       | -       |
| Batch Size            | 32     | 16     | 16      | 32      | 16      |
| Learning Rate         | 0.001  | 0.001  | 0.001   | 0.001   | 0.0005  |
| # Epochs              | 2000   | 200    | 200     | 200     | 100     |
| # Warmup Epochs       | 50     | 5      | 5       | 5       | 5       |
| Weight Decay          | 1e-5   | 1e-5   | 1e-5    | 1e-5    | 1e-5    |
| # Parameters          | 477,241| 118,990| 115,450 | 511,829 | 497,594 |
| Time (epoch)          | 9.4s   | 56.8s  | 46.3s   | 18.5s   | 20.5s   |

## Table 11. Hyperparameter settings of GIN+ on LRGB and OGB datasets.

| Hyperparameter        | Peptides-func | Peptides-struct | PascalVOC-SP | COCO-SP | MalNet-Tiny | ogbg-molhiv | ogbg-molpcba | ogbg-ppa | ogbg-code2 |
|-----------------------|---------------|-----------------|--------------|---------|-------------|-------------|--------------|----------|------------|
| # GNN Layers          | 3             | 5               | 16           | 16      | 5           | 3           | 16           | 5        | 4          |
| Edge Feature Module   | True          | True            | True         | True    | True        | True        | True         | True     | True       |
| Normalization         | BN            | BN              | BN           | BN      | BN          | BN          | BN           | BN       | BN         |
| Dropout               | 0.2           | 0.2             | 0.1          | 0.0     | 0.0         | 0.0         | 0.3          | 0.15     | 0.1        |
| Residual Connections  | True          | True            | True         | True    | True        | True        | True         | False    | True       |
| FFN                   | False         | False           | True         | True    | True        | False       | True         | True     | True       |
| PE                    | RWSE-32       | RWSE-32         | RWSE-32      | False   | False       | RWSE-20     | RWSE-16      | False    | False      |
| Hidden Dim            | 240           | 200             | 70           | 70      | 130         | 256         | 300          | 512      | 512        |
| Graph Pooling         | mean          | mean            | â            | â       | max         | mean        | mean         | mean     | mean       |
| Batch Size            | 16            | 32              | 50           | 50      | 16          | 32          | 512          | 32       | 32         |
| Learning Rate         | 0.0005        | 0.001           | 0.001        | 0.001   | 0.0005      | 0.0001      | 0.0005       | 0.0003   | 0.0001     |
| # Epochs              | 300           | 250             | 200          | 300     | 150         | 100         | 100          | 300      | 30         |
| # Warmup Epochs       | 5             | 5               | 10           | 10      | 10          | 5           | 5            | 10       | 2          |
| Weight Decay          | 0.0           | 0.0             | 0.0          | 0.0     | 1e-5        | 1e-5        | 1e-5         | 1e-5     | 1e-6       |
| # Parameters          | 506,126       | 518,127         | 486,039      | 487,491 | 514,545     | 481,433     | 8,774,720    | 8,173,605| 24,338,354 |
| Time (epoch)          | 7.4s          | 6.1s            | 14.8s        | 169.2s  | 5.9s        | 10.9s       | 89.2s        | 213.9s   | 489.8s     |

# GatedGCN

## Table 12. Hyperparameter settings of GatedGCN+ on benchmarks from (Dwivedi et al., 2023).

| Hyperparameter        | ZINC   | MNIST  | CIFAR10 | PATTERN | CLUSTER |
|-----------------------|--------|--------|---------|---------|---------|
| # GNN Layers          | 9      | 10     | 10      | 12      | 16      |
| Edge Feature Module   | True   | True   | True    | True    | True    |
| Normalization         | BN     | BN     | BN      | BN      | BN      |
| Dropout               | 0.05   | 0.05   | 0.15    | 0.2     | 0.2     |
| Residual Connections  | True   | True   | True    | True    | True    |
| FFN                   | True   | True   | True    | True    | True    |
| PE                    | RWSE-20| False  | False   | RWSE-32 | RWSE-20 |
| Hidden Dim            | 70     | 35     | 35      | 64      | 56      |
| Graph Pooling         | sum    | mean   | mean    | -       | -       |
| Batch Size            | 32     | 16     | 16      | 32      | 16      |
| Learning Rate         | 0.001  | 0.001  | 0.001   | 0.0005  | 0.0005  |
| # Epochs              | 2000   | 200    | 200     | 200     | 100     |
| # Warmup Epochs       | 50     | 5      | 5       | 5       | 5       |
| Weight Decay          | 1e-5   | 1e-5   | 1e-5    | 1e-5    | 1e-5    |
| # Parameters          | 413,355| 118,940| 116,490 | 466,001 | 474,574 |
| Time (epoch)          | 10.5s  | 137.9s | 115.0s  | 32.6s   | 34.1s   |

## Table 13. Hyperparameter settings of GatedGCN+ on LRGB and OGB datasets.

| Hyperparameter        | Peptides-func | Peptides-struct | PascalVOC-SP | COCO-SP | MalNet-Tiny | ogbg-molhiv | ogbg-molpcba | ogbg-ppa | ogbg-code2 |
|-----------------------|---------------|-----------------|--------------|---------|-------------|-------------|--------------|----------|------------|
| # GNN Layers          | 5             | 4               | 12           | 20      | 6           | 3           | 10           | 4        | 5          |
| Edge Feature Module   | True          | True            | True         | True    | True        | True        | True         | True     | True       |
| Normalization         | BN            | BN              | BN           | BN      | BN          | BN          | BN           | BN       | BN         |
| Dropout               | 0.05          | 0.2             | 0.15         | 0.05    | 0.0         | 0.0         | 0.2          | 0.15     | 0.2        |
| Residual Connections  | False         | True            | True         | True    | True        | True        | True         | True     | True       |
| FFN                   | False         | False           | False        | True    | True        | False       | True         | False    | True       |
| PE                    | RWSE-32       | RWSE-32         | RWSE-32      | False   | False       | RWSE-20     | RWSE-16      | False    | False      |
| Hidden Dim            | 135           | 145             | 95           | 52      | 100         | 256         | 256          | 512      | 512        |
| Graph Pooling         | mean          | mean            | -            | -       | max         | mean        | mean         | mean     | mean       |
| Batch Size            | 16            | 32              | 32           | 50      | 16          | 32          | 512          | 32       | 32         |
| Learning Rate         | 0.0005        | 0.001           | 0.001        | 0.001   | 0.0005      | 0.0001      | 0.0005       | 0.0003   | 0.0001     |
| # Epochs              | 300           | 300             | 200          | 300     | 150         | 100         | 100          | 300      | 30         |
| # Warmup Epochs       | 5             | 5               | 10           | 10      | 10          | 5           | 5            | 10       | 2          |
| Weight Decay          | 0.0           | 0.0             | 0.0          | 0.0     | 1e-5        | 1e-5        | 1e-5         | 1e-5     | 1e-6       |
| # Parameters          | 521,141       | 492,897         | 559,094      | 508,589 | 550,905     | 1,076,633   | 6,016,860    | 5,547,557| 29,865,906 |
| Time (epoch)          | 6.9s          | 6.6s            | 12.5s        | 162.5s  | 6.6s        | 16.3s       | 91.4s        | 178.2s   | 476.3s     |