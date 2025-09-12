"""File to do the Mixture of Experts models."""

import copy

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool

from models.graph_model import GNN, UnitaryGCN

"""
class MoE(nn.Module):
    def __init__(self, args):
        super().__init__()
        assert hasattr(args, "layer_types") and len(args.layer_types) == 2, \
            "args.layer_types must be a list of two GNN type strings"
        self.args = args

        # Instantiate two experts by deep-copying args and overriding layer_type
        self.experts = nn.ModuleList()
        for lt in args.layer_types:
            ex_args = copy.deepcopy(args)
            ex_args.layer_type = lt
            # The GNN constructor will consume:
            #   input_dim, hidden_layers, output_dim, num_relations,
            #   mlp, dropout, last_layer_fa, layer_type
            if lt == "Unitary":
                self.experts.append(UnitaryGCN(ex_args))
            else:
                self.experts.append(GNN(ex_args))

        # Router: graph-pooled features → 2 logits
        self.router = nn.Sequential(
            nn.Linear(args.input_dim, args.hidden_layers[0] if args.hidden_layers else args.input_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_layers[0] if args.hidden_layers else args.input_dim, len(args.layer_types))
        )

    def forward(self, graph):
        # 1) Pool raw node features to graph vectors
        pooled = global_mean_pool(graph.x.float(), graph.batch)  # [B, input_dim]

        # 2) Compute routing weights
        logits = self.router(pooled)              # [B, 2]
        weights = F.softmax(logits, dim=-1)       # [B, 2]

        # 3) Compute expert outputs
        y0 = self.experts[0](graph)               # [B, output_dim]
        y1 = self.experts[1](graph)               # [B, output_dim]

        # 4) Weighted sum of experts
        w0 = weights[:, 0].unsqueeze(-1)          # [B,1]
        w1 = weights[:, 1].unsqueeze(-1)          # [B,1]
        y = w0 * y0 + w1 * y1                     # [B, output_dim]

        return y    
"""

################################################################################
######################### Routers Networks #####################################
################################################################################

# These basically compute the weights assigned to each expert's prediction


class MLPRouter(nn.Module):
    """MLP-based router for selecting experts in mixture of experts models.

    Uses global mean pooling to create graph-level representations, then applies
    a multi-layer perceptron to compute expert selection logits.
    """

    def __init__(self, input_dim: int, hidden_layers: list[int], num_experts: int):
        """Initialize the MLP router.

        Args:
            input_dim:
                Dimension of input node features
            hidden_layers:
                List of hidden layer dimensions for the MLP
            num_experts:
                Number of experts to route between
        """
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, num_experts))
        self.net = nn.Sequential(*layers)

    def forward(self, graph):
        z = global_mean_pool(graph.x.float(), graph.batch)
        return self.net(z)


class GNNRouter(nn.Module):
    """GNN-based router for selecting experts in mixture of experts models.

    Uses graph neural network layers to process node features, then applies
    global mean pooling and an MLP to compute expert selection logits.
    """

    def __init__(
        self,
        input_dim: int,
        layer_type: str,
        hidden_dim: int,
        depth: int,
        dropout: float,
        num_experts: int,
    ):
        """Initialize the GNN router.

        Args:
            input_dim:
                Dimension of input node features
            layer_type:
                Type of GNN layer to use ('GCN' or 'GIN')
            hidden_dim:
                Hidden dimension for GNN layers
            depth:
                Number of GNN layers
            dropout:
                Dropout probability
            num_experts:
                Number of experts to route between
        """
        super().__init__()
        assert layer_type in ("GCN", "GIN"), "GNNRouter supports only 'GCN' or 'GIN'"
        self.layer_type = layer_type
        self.dropout = nn.Dropout(dropout)
        self.act_fn = nn.ReLU()

        # Message-passing layers
        dims = [input_dim] + [hidden_dim] * depth
        self.layers = nn.ModuleList(
            [self._get_conv(dims[i], dims[i + 1]) for i in range(depth)]
        )

        # 2-layer MLP for output logits
        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_experts),
        )

    def _get_conv(self, in_feat: int, out_feat: int):
        """Create a graph convolution layer based on layer type.

        Args:
            in_feat:
                Input feature dimension
            out_feat:
                Output feature dimension

        Returns:
            conv_layer:
                GCNConv or GINConv layer
        """
        if self.layer_type == "GCN":
            return GCNConv(in_feat, out_feat)
        # GINConv requires sequential MLP inside
        return GINConv(
            nn.Sequential(
                nn.Linear(in_feat, out_feat),
                nn.BatchNorm1d(out_feat),
                nn.ReLU(),
                nn.Linear(out_feat, out_feat),
            )
        )

    def forward(self, graph):
        """Compute expert selection logits for input graphs.

        Args:
            graph:
                PyTorch Geometric data object with node features x, edge indices, and batch

        Returns:
            logits:
                Expert selection logits of shape [batch_size, num_experts]
        """
        x = graph.x.float()
        edge_index = graph.edge_index
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index)
            if i < len(self.layers) - 1:
                x = self.act_fn(x)
                x = self.dropout(x)
        z = global_mean_pool(x, graph.batch)
        logits = self.mlp_out(z)
        return logits


class Router(nn.Module):
    """Unified router interface that supports both MLP and GNN routing strategies."""

    def __init__(self, router_type: str, input_dim: int, num_experts: int, args):
        """Initialize the router with specified type and configuration.

        Args:
            router_type:
                Type of router to use ('MLP' or 'GNN')
            input_dim:
                Dimension of input node features
            num_experts:
                Number of experts to route between
            args:
                Configuration object containing router-specific parameters
        """
        super().__init__()
        if router_type == "MLP":
            self.model = MLPRouter(input_dim, args.router_hidden_layers, num_experts)
        elif router_type == "GNN":
            hidden_dim = 64
            self.model = GNNRouter(
                input_dim=input_dim,
                layer_type=args.router_layer_type,
                hidden_dim=hidden_dim,
                depth=args.router_depth,
                dropout=args.router_dropout,
                num_experts=num_experts,
            )
        else:
            raise ValueError(f"Unknown router_type: {router_type}")

    def forward(self, graph):
        """Compute expert selection logits for input graphs.

        Args:
            graph:
                PyTorch Geometric data object

        Returns:
            logits:
                Expert selection logits of shape [batch_size, num_experts]
        """
        return self.model(graph)


################################################################################
######################### MoE Models ###########################################
################################################################################

# This is basically ROuters outputs + Experts outputs


class MoE(nn.Module):
    """Mixture of Experts model for graph neural networks.

    Combines multiple expert GNN models using a learned routing mechanism
    to dynamically select and weight expert contributions for each input graph.
    """

    def __init__(self, args):
        """Initialize the Mixture of Experts model.

        Args:
            args:
                Configuration object with attributes:
                - layer_types: list of two expert type strings
                - router_type: type of router ('MLP' or 'GNN')
                - input_dim: dimension of input node features
                - other expert-specific and router-specific parameters
        """
        super().__init__()
        assert (
            hasattr(args, "layer_types") and len(args.layer_types) == 2
        ), "args.layer_types must be a list of two expert type strings"
        self.args = args

        # Instantiate experts
        self.experts = nn.ModuleList()
        for lt in args.layer_types:
            ex_args = copy.deepcopy(args)
            ex_args.layer_type = lt
            self.experts.append(
                UnitaryGCN(ex_args) if lt == "Unitary" else GNN(ex_args)
            )

        # Instantiate router
        self.router = Router(
            router_type=args.router_type,
            input_dim=args.input_dim,
            num_experts=len(args.layer_types),
            args=args,
        )

    def forward(self, graph):
        """Forward pass through the mixture of experts model.

        Args:
            graph:
                PyTorch Geometric data object with node features and graph structure

        Returns:
            y:
                Weighted combination of expert outputs, shape [batch_size, output_dim]
        """
        logits = self.router(
            graph
        )  # [B, num_experts] # note that _call_ calls the forward method
        weights = F.softmax(logits, dim=-1)  # [B, num_experts]
        outs = [expert(graph) for expert in self.experts]  # list of [B, output_dim]
        y = sum(
            weights[:, i].unsqueeze(-1) * outs[i] for i in range(len(outs))
        )  # [B, output_dim]
        return y


class MoE_E(nn.Module):
    """Mixture of Experts model with feature masking for the second expert.

    Similar to MoE but applies feature masking to the input of the second expert,
    zeroing out the last 5 dimensions of node features before processing.
    """

    def __init__(self, args):
        """Initialize the MoE_E model with feature masking.

        Args:
            args:
                Configuration object with attributes:
                - layer_types: list of two expert type strings
                - router_type: type of router ('MLP' or 'GNN')
                - input_dim: dimension of input node features
                - other expert-specific and router-specific parameters
        """
        super().__init__()
        assert (
            hasattr(args, "layer_types") and len(args.layer_types) == 2
        ), "args.layer_types must be a list of two expert type strings"
        self.args = args

        # Instantiate experts
        self.experts = nn.ModuleList()

        # Instantiate experts
        self.experts = nn.ModuleList()
        for lt in args.layer_types:
            ex_args = copy.deepcopy(args)
            ex_args.layer_type = lt
            self.experts.append(
                UnitaryGCN(ex_args) if lt == "Unitary" else GNN(ex_args)
            )

        # Instantiate router
        self.router = Router(
            router_type=args.router_type,
            input_dim=args.input_dim,
            num_experts=len(args.layer_types),
            args=args,
        )

    def forward(self, graph):
        """Forward pass with feature masking for the second expert.

        Args:
            graph:
                PyTorch Geometric data object with node features and graph structure

        Returns:
            y:
                Weighted combination of expert outputs, shape [batch_size, output_dim]
        """
        logits = self.router(graph)  # [B, num_experts]
        weights = F.softmax(logits, dim=-1)  # [B, num_experts]

        # mask that preserves the first F-5 dims and zeros the last 5
        projection = torch.cat(
            [
                torch.ones(graph.x.size(-1) - 5, device=graph.x.device),
                torch.zeros(5, device=graph.x.device),
            ]
        )  # shape: [F]

        outs = []
        outs.append(self.experts[0](graph))

        graph.x = graph.x * projection  # zero out last five dims
        outs.append(self.experts[1](graph))

        y = sum(
            weights[:, i].unsqueeze(-1) * outs[i] for i in range(len(outs))
        )  # [B, output_dim]
        return y
