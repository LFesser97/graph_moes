"""GNN-based router for Mixture of Experts."""

from typing import Union

import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool


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
    ) -> None:
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

    def _get_conv(self, in_feat: int, out_feat: int) -> Union[GCNConv, GINConv]:
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

    def forward(self, graph: Union[Data, Batch]) -> torch.Tensor:
        """Compute expert selection logits for input graphs.

        Args:
            graph:
                PyTorch Geometric data object with node features x, edge indices, and batch

        Returns:
            logits:
                Expert selection logits of shape [batch_size, num_experts]
                Each row contains logits for one graph, one per expert.
                These will be softmaxed to get routing weights (each row sums to 1.0).
        """
        x = graph.x.float()
        edge_index = graph.edge_index
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index)
            if i < len(self.layers) - 1:
                x = self.act_fn(x)
                x = self.dropout(x)
        z = global_mean_pool(x, graph.batch)  # [batch_size, hidden_dim]
        logits = self.mlp_out(
            z
        )  # [batch_size, num_experts] - one logit per expert per graph
        return logits
