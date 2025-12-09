https://arxiv.org/abs/2003.00982: Benchmarking Graph Neural Networks

# MNIST and CIFAR10 Superpixel Graph Datasets

Official Repository: The graph versions of MNIST and CIFAR10 (used in Dwivedi et al. 2020) are provided in the Benchmarking Graph Neural Networks codebase (https://github.com/graphdeeplearning/benchmarking-gnns). In these datasets, each image is converted into a graph using SLIC superpixels (each node represents a superpixel region) with k-NN connectivity. Both datasets are available via PyTorch Geometric’s GNNBenchmarkDataset (https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/gnn_benchmark_dataset.html#:~:text=root_url%20%3D%20%27https%3A%2F%2Fdata.pyg.org%2Fdatasets%2Fbenchmarking,root_url%7D%2FTSP_v2.zip%27%2C%20%27CSL%27%3A%20%27https%3A%2F%2Fwww.dropbox.com%2Fs%2Frnbkp5ubgk82ocu%2FCSL.zip%3Fdl%3D1%27%2C) or via the original benchmarking repository. The standard splits are the same as the image datasets (e.g. 60k training and 10k test graphs for MNIST).



# PATTERN Graph Dataset 

Official Repository: The PATTERN dataset is a synthetic graph dataset introduced by Dwivedi et al. (Benchmarking GNNs) for node classification. Graphs in PATTERN are generated using a Stochastic Block Model (SBM) to create community structures. The official implementation is included in the Benchmarking GNNs GitHub repo (which provides the data generator or pre-saved dataset). PyTorch Geometric also provides PATTERN via the GNNBenchmarkDataset class, downloading it as PATTERN_v2.zip from the benchmarking dataset repository. The dataset consists of 14,000 graphs (10k train, 2k validation, 2k test) each with ~118 nodes on average, and nodes have 3-dimensional feature vectors with a binary label to predict

##  Classification

All three datasets should be used for CLASSIFICATION:

MNIST Superpixels: Graph classification (digit recognition, 10 classes: 0-9)
CIFAR-10 Superpixels: Graph classification (object recognition, 10 classes: airplane, car, bird, etc.)
PATTERN: Graph classification (synthetic dataset with binary node classification, but used as graph classification in benchmarks)
