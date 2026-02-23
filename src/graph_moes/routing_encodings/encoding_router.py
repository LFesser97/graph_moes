"""Router for selecting encodings in EncodingMoE."""

from typing import Any, Union

import torch
from torch import nn
from torch_geometric.data import Batch, Data

from graph_moes.moes.routers.gnn_router import GNNRouter
from graph_moes.moes.routers.mlp_router import MLPRouter


class EncodingRouter(nn.Module):
    """Router for selecting which encoding to append to graph features.

    Similar to MoE routers but designed for encoding selection. Uses either
    MLP or GNN to route between different encoding options.
    """

    def __init__(
        self, router_type: str, input_dim: int, num_encodings: int, args: Any
    ) -> None:
        """Initialize the encoding router.

        Args:
            router_type:
                Type of router to use ('MLP' or 'GNN')
            input_dim:
                Dimension of input node features (base features before encoding)
            num_encodings:
                Number of encoding options to route between
            args:
                Configuration object containing router-specific parameters:
                - router_hidden_layers: List[int] (for MLP router)
                - router_layer_type: str (for GNN router, default 'GIN')
                - router_depth: int (for GNN router, default 4)
                - router_dropout: float (for GNN router, default 0.1)
        """
        super().__init__()
        if router_type == "MLP":
            hidden_layers = getattr(args, "router_hidden_layers", [64, 64, 64])
            self.model: Union[MLPRouter, GNNRouter] = MLPRouter(
                input_dim, hidden_layers, num_encodings
            )
        elif router_type == "GNN":
            hidden_dim = 64
            self.model = GNNRouter(
                input_dim=input_dim,
                layer_type=getattr(args, "router_layer_type", "GIN"),
                hidden_dim=hidden_dim,
                depth=getattr(args, "router_depth", 4),
                dropout=getattr(args, "router_dropout", 0.1),
                num_experts=num_encodings,
            )
        else:
            raise ValueError(f"Unknown router_type: {router_type}")

    def forward(self, graph: Union[Data, Batch]) -> torch.Tensor:
        """Compute encoding selection logits for input graphs.

        Args:
            graph:
                PyTorch Geometric data object with base node features (before encoding)

        Returns:
            logits:
                Encoding selection logits of shape [batch_size, num_encodings]
                Each row contains logits for one graph, one per encoding option.
                These will be softmaxed to get routing weights (each row sums to 1.0).
        """
        result = self.model(graph)
        assert isinstance(result, torch.Tensor), "Router must return a Tensor"
        return result
