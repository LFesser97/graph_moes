"""Graph Neural Network Model Implementations for Graph Learning Tasks.

This module provides various Graph Neural Network (GNN) architectures and custom
convolution layers for graph classification and regression tasks. The implementations
support different types of graph convolutions, attention mechanisms, and pooling
strategies.

Classes:
    RGATConv: Relational Graph Attention Network convolution layer
    RGINConv: Relational Graph Isomorphism Network convolution layer
    GNN: Configurable multi-layer GNN supporting various convolution types
    GPS: Graph PolyNomial Sampling model with Performer attention
    UnitaryGCN: Unitary Graph Convolutional Network using complex-valued layers
    OrthogonalGCN: Orthogonal Graph Convolutional Network with real-valued constraints
    GatedGCNConv: Gated Graph Convolutional Network layer with edge features
    RedrawProjection: Utility class for Performer attention projection matrix updates
"""

import torch
import torch.nn as nn
from measure_smoothing import dirichlet_normalized
from torch.nn import ModuleList, Dropout, ReLU
from torch_geometric.nn import (
    GCNConv,
    RGCNConv,
    SAGEConv,
    GatedGraphConv,
    GINConv,
    FiLMConv,
    global_mean_pool,
    GATConv,
    SuperGATConv,
    global_max_pool,
    GPSConv,
    GINEConv,
    global_add_pool,
)


import argparse
from typing import Any, Dict, Optional

import torch.nn.functional as F
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)

import torch_geometric.transforms as T
from models.performer import PerformerAttention

from models.layers import TaylorGCNConv, ComplexGCNConv
from models.complex_valued_layers import UnitaryGCNConvLayer
from models.real_valued_layers import OrthogonalGCNConvLayer

import torch
import torch.nn.functional as F
from torch import nn


class RGATConv(torch.nn.Module):
    def __init__(self, in_features, out_features, num_relations):
        super(RGATConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.self_loop_conv = torch.nn.Linear(in_features, out_features)
        convs = []
        for i in range(self.num_relations):
            convs.append(GATConv(in_features, out_features))
        self.convs = ModuleList(convs)

    def forward(self, x, edge_index, edge_type):
        x_new = self.self_loop_conv(x)
        for i, conv in enumerate(self.convs):
            rel_edge_index = edge_index[:, edge_type == i]
            x_new += conv(x, rel_edge_index)
        return x_new


class RGINConv(torch.nn.Module):
    def __init__(self, in_features, out_features, num_relations):
        super(RGINConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.self_loop_conv = torch.nn.Linear(in_features, out_features)
        convs = []
        for i in range(self.num_relations):
            convs.append(
                GINConv(
                    nn.Sequential(
                        nn.Linear(in_features, out_features),
                        nn.BatchNorm1d(out_features),
                        nn.ReLU(),
                        nn.Linear(out_features, out_features),
                    )
                )
            )
        self.convs = ModuleList(convs)

    def forward(self, x, edge_index, edge_type):
        x_new = self.self_loop_conv(x)
        for i, conv in enumerate(self.convs):
            rel_edge_index = edge_index[:, edge_type == i]
            x_new += conv(x, rel_edge_index)
        return x_new


class GNN(torch.nn.Module):
    def __init__(self, args):
        super(GNN, self).__init__()
        self.args = args
        self.final_layer = args.mlp
        self.num_relations = args.num_relations
        self.layer_type = args.layer_type
        if self.final_layer:
            num_features = [args.input_dim] + list(args.hidden_layers)
        else:
            num_features = (
                [args.input_dim] + list(args.hidden_layers) + [args.output_dim]
            )
        self.num_layers = len(num_features) - 1
        layers = []
        for i, (in_features, out_features) in enumerate(
            zip(num_features[:-1], num_features[1:])
        ):
            layers.append(self.get_layer(in_features, out_features))
        self.layers = ModuleList(layers)
        self.dropout = Dropout(p=args.dropout)
        self.act_fn = ReLU()

        if self.args.last_layer_fa:
            if self.layer_type == "R-GCN" or self.layer_type == "GCN":
                self.last_layer_transform = torch.nn.Linear(
                    self.args.hidden_dim, self.args.output_dim
                )
            elif self.layer_type == "R-GIN" or self.layer_type == "GIN":
                self.last_layer_transform = nn.Sequential(
                    nn.Linear(self.args.hidden_dim, self.args.hidden_dim),
                    nn.BatchNorm1d(self.args.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.args.hidden_dim, self.args.output_dim),
                )
            else:
                raise NotImplementedError

        self.mlp = Sequential(
            Linear(self.args.hidden_dim, self.args.hidden_dim // 2),
            ReLU(),
            Linear(self.args.hidden_dim // 2, self.args.hidden_dim // 4),
            ReLU(),
            Linear(self.args.hidden_dim // 4, self.args.output_dim),
        )

    def get_layer(self, in_features, out_features):
        if self.layer_type == "GCN":
            return GCNConv(in_features, out_features)
        elif self.layer_type == "R-GCN":
            return RGCNConv(in_features, out_features, self.num_relations)
        elif self.layer_type == "R-GAT":
            return RGATConv(in_features, out_features, self.num_relations)
        elif self.layer_type == "R-GIN":
            return RGINConv(in_features, out_features, self.num_relations)
        elif self.layer_type == "GIN":
            return GINConv(
                nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.BatchNorm1d(out_features),
                    nn.ReLU(),
                    nn.Linear(out_features, out_features),
                )
            )
        elif self.layer_type == "SAGE":
            return SAGEConv(in_features, out_features)
        elif self.layer_type == "FiLM":
            return FiLMConv(in_features, out_features)
        elif self.layer_type == "MLP":  # <<< added new branch
            # <<< begin added lines for MLP layer type
            return nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.ReLU(),
            )  # <<< end added lines
        else:
            raise ValueError(f"Unknown layer_type: {self.layer_type}")

    def forward(self, graph, measure_dirichlet=False):
        x, edge_index, ptr, batch = graph.x, graph.edge_index, graph.ptr, graph.batch
        x = x.float()
        for i, layer in enumerate(self.layers):
            # MLP layers ignore graph structure
            if self.layer_type == "MLP":  # <<< added condition
                x_new = layer(x)  # <<< changed: MLP uses only x
            else:
                if self.layer_type in ["R-GCN", "R-GAT", "R-GIN", "FiLM"]:
                    x_new = layer(x, edge_index, edge_type=graph.edge_type)
                else:
                    x_new = layer(x, edge_index)
            if i != self.num_layers - 1:
                x_new = self.act_fn(x_new)
                x_new = self.dropout(x_new)
            if i == self.num_layers - 1 and self.args.last_layer_fa:
                combined_values = global_mean_pool(x, batch)
                # combined_values = global_max_pool(x, batch)
                # print("Using global max pooling."
                combined_values = self.last_layer_transform(combined_values)
                if self.layer_type in ["R-GCN", "R-GIN"]:
                    x_new += combined_values[batch]
                else:
                    x_new = combined_values[batch]
            x = x_new
        if measure_dirichlet:
            energy = dirichlet_normalized(
                x.cpu().numpy(), graph.edge_index.cpu().numpy()
            )
            return energy
        x = global_mean_pool(x, batch)
        if self.final_layer:
            return self.mlp(x)
        else:
            return x


class GPS(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        pe_dim = 20
        channels = args.hidden_dim
        input_dim = args.input_dim
        num_layers = len(list(args.hidden_layers)) + 1
        attn_type = "performer"
        output_dim = args.output_dim

        # self.node_emb = Linear(1, channels - pe_dim)
        self.node_emb = Linear(input_dim, channels)
        self.pe_lin = Linear(20, pe_dim)
        self.pe_norm = BatchNorm1d(20)
        self.edge_emb = Embedding(4, channels)

        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            conv = GPSConv(channels, GINConv(nn), heads=2)
            self.convs.append(conv)

        self.mlp = Sequential(
            Linear(channels, channels // 2),
            ReLU(),
            Linear(channels // 2, channels // 4),
            ReLU(),
            Linear(channels // 4, output_dim),
        )

        self.redraw_projection = RedrawProjection(
            self.convs, redraw_interval=1000 if attn_type == "performer" else None
        )

    # def forward(self, x, pe, edge_index, edge_attr, batch):
    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        # x_pe = self.pe_norm(pe)
        # x = torch.cat((self.node_emb(x.squeeze(-1)), self.pe_lin(x_pe)), 1)
        # x = torch.cat((self.node_emb(x), self.pe_lin(x_pe)), 1)
        x = self.node_emb(x.float())
        # edge_attr = self.edge_emb(edge_attr)

        for conv in self.convs:
            x = conv(x, edge_index, batch)  # , edge_attr=edge_attr)
        x = global_add_pool(x, batch)
        return F.log_softmax(self.mlp(x), dim=1)


class RedrawProjection:
    def __init__(self, model: torch.nn.Module, redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module
                for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1


class UnitaryGCN(nn.Module):
    def __init__(self, args):
        super(UnitaryGCN, self).__init__()
        self.conv_layers = nn.ModuleList()
        input_dim = args.input_dim
        hidden_dim = args.hidden_dim
        output_dim = args.output_dim
        num_layers = args.num_layers
        hidden_layer_dim = args.hidden_dim
        self.T = 20
        self.dropout = Dropout(p=args.dropout)
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(UnitaryGCNConvLayer(input_dim, hidden_dim))
        for _ in range(num_layers):
            self.conv_layers.append(
                UnitaryGCNConvLayer(hidden_dim, hidden_dim, use_hermitian=True)
            )
        self.hidden_layer = nn.Linear(hidden_dim, hidden_layer_dim)
        self.output_layer = nn.Linear(hidden_layer_dim, output_dim)
        self.reset_parameters()

    def forward(self, data):
        for conv in self.conv_layers:
            data = conv(data)
        x = global_mean_pool(data.x.real, data.batch)  # Global pooling over nodes
        x = F.relu(self.hidden_layer(x))  # Hidden layer with ReLU activation
        x = self.output_layer(x)  # Output layer
        return x

    def reset_parameters(self):
        pass


class OrthogonalGCN(nn.Module):
    def __init__(self, args):
        super(OrthogonalGCN, self).__init__()
        self.conv_layers = nn.ModuleList()
        input_dim = args.input_dim
        hidden_dim = 64
        output_dim = args.output_dim
        num_layers = 4
        hidden_layer_dim = 64
        self.T = 20
        self.dropout = Dropout(p=args.dropout)
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(OrthogonalGCNConvLayer(input_dim, hidden_dim))
        for _ in range(num_layers):
            self.conv_layers.append(
                OrthogonalGCNConvLayer(hidden_dim, hidden_dim, use_hermitian=True)
            )
        self.hidden_layer = nn.Linear(hidden_dim, hidden_layer_dim)
        self.output_layer = nn.Linear(hidden_layer_dim, output_dim)
        self.reset_parameters()

    def forward(self, data):
        for conv in self.conv_layers:
            data = conv(data)
        x = global_mean_pool(data.x.real, data.batch)  # Global pooling over nodes
        x = F.relu(self.hidden_layer(x))  # Hidden layer with ReLU activation
        x = self.output_layer(x)  # Output layer
        return x

    def reset_parameters(self):
        pass


# Adding GatedGNN:
# Source:
# https://www.dgl.ai/dgl_docs/_modules/dgl/nn/pytorch/conv/gatedgcnconv.html
class GatedGCNConv(nn.Module):
    """Gated graph convolutional layer from `Benchmarking Graph Neural Networks
    <https://arxiv.org/abs/2003.00982>`__

    .. math::
        e_{ij}^{l+1}=D^l h_{i}^{l}+E^l h_{j}^{l}+C^l e_{ij}^{l}

        norm_{ij}=\Sigma_{j\in N_{i}} \sigma\left(e_{ij}^{l+1}\right)+\varepsilon

        \hat{e}_{ij}^{l+1}=\sigma(e_{ij}^{l+1}) / norm_{ij}

        h_{i}^{l+1}=A^l h_{i}^{l}+\Sigma_{j \in N_{i}} \hat{e}_{ij}^{l+1} \odot B^l h_{j}^{l}

    where :math:`h_{i}^{l}` is node :math:`i` feature of layer :math:`l`,
    :math:`e_{ij}^{l}` is edge :math:`ij` feature of layer :math:`l`,
    :math:`\sigma` is sigmoid function, :math:`\varepsilon` is a small fixed constant
    for numerical stability, :math:`A^l, B^l, C^l, D^l, E^l` are linear layers.

    Parameters
    ----------
    input_feats : int
        Input feature size; i.e, the number of dimensions of :math:`h_{i}^{l}`.
    edge_feats: int
        Edge feature size; i.e., the number of dimensions of :math:`e_{ij}^{l}`.
    output_feats : int
        Output feature size; i.e., the number of dimensions of :math:`h_{i}^{l+1}`.
    dropout : float, optional
        Dropout rate on node and edge feature. Default: ``0``.
    batch_norm : bool, optional
        Whether to include batch normalization on node and edge feature. Default: ``True``.
    residual : bool, optional
        Whether to include residual connections. Default: ``True``.
    activation : callable activation function/layer or None, optional
        If not None, apply an activation function to the updated node features.
        Default: ``F.relu``.

    Example
    -------
    >>> import dgl
    >>> import torch as th
    >>> import torch.nn.functional as F
    >>> from dgl.nn import GatedGCNConv

    >>> num_nodes, num_edges = 8, 30
    >>> graph = dgl.rand_graph(num_nodes,num_edges)
    >>> node_feats = th.rand(num_nodes, 20)
    >>> edge_feats = th.rand(num_edges, 12)
    >>> gatedGCN = GatedGCNConv(20, 12, 20)
    >>> new_node_feats, new_edge_feats = gatedGCN(graph, node_feats, edge_feats)
    >>> new_node_feats.shape, new_edge_feats.shape
    (torch.Size([8, 20]), torch.Size([30, 20]))
    """

    def __init__(
        self,
        input_feats,
        edge_feats,
        output_feats,
        dropout=0,
        batch_norm=True,
        residual=True,
        activation=F.relu,
    ):
        super(GatedGCNConv, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = batch_norm
        self.residual = residual

        if input_feats != output_feats or edge_feats != output_feats:
            self.residual = False

        # Linearly transform the node features.
        self.A = nn.Linear(input_feats, output_feats, bias=True)
        self.B = nn.Linear(input_feats, output_feats, bias=True)
        self.D = nn.Linear(input_feats, output_feats, bias=True)
        self.E = nn.Linear(input_feats, output_feats, bias=True)

        # Linearly transform the edge features.
        self.C = nn.Linear(edge_feats, output_feats, bias=True)

        # Batch normalization on the node/edge features.
        self.bn_node = nn.BatchNorm1d(output_feats)
        self.bn_edge = nn.BatchNorm1d(output_feats)

        self.activation = activation

    def forward(self, graph, feat, edge_feat):
        """

        Description
        -----------
        Compute gated graph convolution layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`N`
            is the number of nodes of the graph and :math:`D_{in}` is the
            input feature size.
        edge_feat : torch.Tensor
            The input edge feature of shape :math:`(E, D_{edge})`,
            where :math:`E` is the number of edges and :math:`D_{edge}`
            is the size of the edge features.

        Returns
        -------
        torch.Tensor
            The output node feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is the output feature size.
        torch.Tensor
            The output edge feature of shape :math:`(E, D_{out})` where :math:`D_{out}`
            is the output feature size.
        """
        with graph.local_scope():
            # For residual connection
            h_in = feat
            e_in = edge_feat

            graph.ndata["Ah"] = self.A(feat)
            graph.ndata["Bh"] = self.B(feat)
            graph.ndata["Dh"] = self.D(feat)
            graph.ndata["Eh"] = self.E(feat)
            graph.edata["Ce"] = self.C(edge_feat)

            graph.apply_edges(fn.u_add_v("Dh", "Eh", "DEh"))

            # Get edge feature
            graph.edata["e"] = graph.edata["DEh"] + graph.edata["Ce"]
            graph.edata["sigma"] = torch.sigmoid(graph.edata["e"])

            graph.update_all(fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h"))
            graph.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
            graph.ndata["h"] = graph.ndata["Ah"] + graph.ndata["sum_sigma_h"] / (
                graph.ndata["sum_sigma"] + 1e-6
            )

            # Result of graph convolution.
            feat = graph.ndata["h"]
            edge_feat = graph.edata["e"]

            # Batch normalization.
            if self.batch_norm:
                feat = self.bn_node(feat)
                edge_feat = self.bn_edge(edge_feat)

            # Non-linear activation.
            if self.activation:
                feat = self.activation(feat)
                edge_feat = self.activation(edge_feat)

            # Residual connection.
            if self.residual:
                feat = h_in + feat
                edge_feat = e_in + edge_feat

            feat = self.dropout(feat)
            edge_feat = self.dropout(edge_feat)

            return feat, edge_feat
