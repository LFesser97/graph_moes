# Gated GNN:

RESIDUAL GATED GRAPH CONVNETS: This is the paper.


Official Implementation: Gated Graph ConvNet (GatedGCN) was proposed by Bresson & Laurent (2017) as Residual Gated Graph ConvNets (ICLR 2018). The authors released an official PyTorch implementation on GitHub: https://github.com/xbresson/spatial_graph_convnets?utm_source=catalyzex.com. This original code (in Jupyter notebooks) demonstrates GatedGCN on tasks like subgraph matching and graph clustering.

Modern Implementations: GatedGCN has since been adopted in popular graph libraries. For example, PyTorch Geometric (PyG) includes a ResGatedGraphConv layer which implements the residual gated convolution from the original paper. Likewise, Deep Graph Library (DGL) provides a GatedGCNConv module following the formulation in the Benchmarking GNNs paper: https://www.dgl.ai/dgl_docs/_modules/dgl/nn/pytorch/conv/gatedgcnconv.html. These implementations incorporate the key ideas of GatedGCN: residual connections, batch normalization, and gating mechanisms on edges or node pairs.

I am using the DGL implementation.