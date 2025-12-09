"""MLP-based router for Mixture of Experts."""

from typing import Union

import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool


class MLPRouter(nn.Module):
    """MLP-based router for selecting experts in mixture of experts models.

    Uses global mean pooling to create graph-level representations, then applies
    a multi-layer perceptron to compute expert selection logits.
    """

    def __init__(
        self, input_dim: int, hidden_layers: list[int], num_experts: int
    ) -> None:
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

    def forward(self, graph: Union[Data, Batch]) -> torch.Tensor:
        """Compute routing logits for each graph in the batch.

        Returns:
            logits: shape [batch_size, num_experts]
                One logit per expert for each graph. These will be softmaxed
                to get routing weights where each row sums to 1.0.
        """
        z = global_mean_pool(graph.x.float(), graph.batch)  # [batch_size, input_dim]
        return self.net(z)  # [batch_size, num_experts]
