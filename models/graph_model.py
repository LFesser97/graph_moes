"""Graph Neural Network Model Implementations for Graph Learning Tasks.

This module provides various Graph Neural Network (GNN) architectures and custom
convolution layers for graph classification and regression tasks. The implementations
support different types of graph convolutions, attention mechanisms, and pooling
strategies.

Classes:
    GNN: Configurable multi-layer GNN supporting various convolution types
    GPS: Graph PolyNomial Sampling model with Performer attention
    UnitaryGCN: Unitary Graph Convolutional Network using complex-valued layers
    OrthogonalGCN: Orthogonal Graph Convolutional Network with real-valued constraints

    RGATConv: Relational Graph Attention Network convolution layer
    RGINConv: Relational Graph Isomorphism Network convolution layer
    ResGatedGraphConv: Gated Graph Convolutional Network layer with edge features
    RedrawProjection: Utility class for Performer attention projection matrix updates

## Layers:
From PyTorch Geometric (imported):
GCNConv - Graph Convolutional Network layer
RGCNConv - Relational GCN (multiple edge types)
SAGEConv - GraphSAGE layer
GINConv - Graph Isomorphism Network layer
FiLMConv - Feature-wise Linear Modulation layer
GPSConv - Graph PolyNomial Sampling layer

Custom Layers (we implemented):
RGATConv - custom Relational Graph Attention layer
RGINConv - custom Relational Graph Isomorphism layer
ResGatedGraphConv - Residual Gated Graph Convolution layer

Pseudo-Layer:
MLP - Multi-Layer Perceptron (ignores graph structure

## MODELS (Complete Architectures)
Main Models:
GNN - The flexible, configurable model that can use any layer type
GPS - model with Performer attention
UnitaryGCN - Complex-valued unitary transformations model
OrthogonalGCN - Real-valued orthogonal transformations model
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

from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Parameter, Sigmoid

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor

from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Parameter, Sigmoid

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor


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
        elif self.layer_type == "GatedGCN":
            return ResGatedGraphConv(in_features, out_features)
        elif self.layer_type == "MLP":
            return nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.ReLU(),
            )
        else:
            raise ValueError(f"Unknown layer_type: {self.layer_type}")

    def forward(self, graph, measure_dirichlet=False):
        x, edge_index, ptr, batch = graph.x, graph.edge_index, graph.ptr, graph.batch
        x = x.float()
        for i, layer in enumerate(self.layers):
            # MLP layers ignore graph structure
            if self.layer_type == "MLP":
                x_new = layer(x)
            elif self.layer_type == "GatedGCN":
                if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
                    x_new = layer(x, edge_index, graph.edge_attr)
                else:
                    x_new = layer(x, edge_index)  # Works without edge attributes
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


##############################


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


class ResGatedGraphConv(MessagePassing):
    r"""The residual gated graph convolutional operator from the
    `"Residual Gated Graph ConvNets" <https://arxiv.org/abs/1711.07553>`_
    paper.

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \eta_{i,j} \odot \mathbf{W}_2 \mathbf{x}_j

    where the gate :math:`\eta_{i,j}` is defined as

    .. math::
        \eta_{i,j} = \sigma(\mathbf{W}_3 \mathbf{x}_i + \mathbf{W}_4
        \mathbf{x}_j)

    with :math:`\sigma` denoting the sigmoid function.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        act (callable, optional): Gating function :math:`\sigma`.
            (default: :meth:`torch.nn.Sigmoid()`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default: :obj:`None`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **inputs:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`        - **outputs:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V_t}|, F_{out})` if bipartite
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        act: Optional[Callable] = Sigmoid(),
        edge_dim: Optional[int] = None,
        root_weight: bool = True,
        bias: bool = True,
        **kwargs,
    ):

        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act
        self.edge_dim = edge_dim
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        edge_dim = edge_dim if edge_dim is not None else 0
        self.lin_key = Linear(in_channels[1] + edge_dim, out_channels)
        self.lin_query = Linear(in_channels[0] + edge_dim, out_channels)
        self.lin_value = Linear(in_channels[0] + edge_dim, out_channels)

        if root_weight:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=False)
        else:
            self.register_parameter("lin_skip", None)

        if bias:
            self.bias = Parameter(Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.lin_skip is not None:
            self.lin_skip.reset_parameters()
        if self.bias is not None:
            zeros(self.bias)

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
    ) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        # In case edge features are not given, we can compute key, query and
        # value tensors in node-level space, which is a bit more efficient:
        if self.edge_dim is None:
            k = self.lin_key(x[1])
            q = self.lin_query(x[0])
            v = self.lin_value(x[0])
        else:
            k, q, v = x[1], x[0], x[0]

        # propagate_type: (k: Tensor, q: Tensor, v: Tensor,
        #                  edge_attr: OptTensor)
        out = self.propagate(edge_index, k=k, q=q, v=v, edge_attr=edge_attr)

        if self.root_weight:
            out = out + self.lin_skip(x[1])

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(
        self, k_i: Tensor, q_j: Tensor, v_j: Tensor, edge_attr: OptTensor
    ) -> Tensor:

        assert (edge_attr is not None) == (self.edge_dim is not None)

        if edge_attr is not None:
            k_i = self.lin_key(torch.cat([k_i, edge_attr], dim=-1))
            q_j = self.lin_query(torch.cat([q_j, edge_attr], dim=-1))
            v_j = self.lin_value(torch.cat([v_j, edge_attr], dim=-1))

        return self.act(k_i + q_j) * v_j
