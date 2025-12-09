"""Unified router interface for Mixture of Experts."""

from typing import Any, Union

import torch
from torch import nn
from torch_geometric.data import Batch, Data

from graph_moes.moes.routers.gnn_router import GNNRouter
from graph_moes.moes.routers.mlp_router import MLPRouter


class Router(nn.Module):
    """Unified router interface that supports both MLP and GNN routing strategies."""

    def __init__(
        self, router_type: str, input_dim: int, num_experts: int, args: Any
    ) -> None:
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

    def forward(self, graph: Union[Data, Batch]) -> torch.Tensor:
        """Compute expert selection logits for input graphs.

        Args:
            graph:
                PyTorch Geometric data object (can be batched)

        Returns:
            logits:
                Expert selection logits of shape [batch_size, num_experts]
                Each row contains logits for one graph, one per expert.
                These will be softmaxed to get routing weights (each row sums to 1.0).
        """
        return self.model(graph)
