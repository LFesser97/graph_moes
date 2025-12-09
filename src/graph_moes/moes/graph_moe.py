"""Mixture of Experts models for graph neural networks."""

import copy
from typing import Any, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch, Data

from graph_moes.architectures.graph_model import GNN, UnitaryGCN
from graph_moes.moes.routers.router import Router

################################################################################
######################### MoE Models ###########################################
################################################################################

# This is basically Routers outputs + Experts outputs


class MoE(nn.Module):
    """Mixture of Experts model for graph neural networks.

    Combines multiple expert GNN models using a learned routing mechanism
    to dynamically select and weight expert contributions for each input graph.
    """

    def __init__(self, args: Any) -> None:
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

    def forward(self, graph, return_weights: bool = False):
        """Forward pass through the mixture of experts model.

        Args:
            graph:
                PyTorch Geometric data object with node features and graph structure
            return_weights:
                If True, also return the routing weights assigned to each expert

        Returns:
            y:
                Weighted combination of expert outputs, shape [batch_size, output_dim]
            weights (optional):
                Routing weights assigned to each expert, shape [batch_size, num_experts]
                Each row contains weights for one graph (sums to 1.0 due to softmax).
                Example: weights[0] = [0.7, 0.3] means graph 0 uses 70% expert 0, 30% expert 1.
                Only returned if return_weights=True
        """
        # Router computes logits for each graph in the batch
        logits = self.router(
            graph
        )  # [B, num_experts] - one logit per expert for each graph in batch
        # Apply softmax to get normalized weights (each row sums to 1.0)
        weights = F.softmax(logits, dim=-1)  # [B, num_experts]
        # Each graph gets its own routing weights, so different graphs can prefer different experts
        outs = [expert(graph) for expert in self.experts]  # list of [B, output_dim]
        # Weighted combination: for each graph, combine expert outputs using its routing weights
        y = sum(
            weights[:, i].unsqueeze(-1) * outs[i] for i in range(len(outs))
        )  # [B, output_dim]
        if return_weights:
            return y, weights
        return y


class MoE_E(nn.Module):
    """Mixture of Experts model with feature masking for the second expert.

    Similar to MoE but applies feature masking to the input of the second expert,
    zeroing out the last 5 dimensions of node features before processing.
    """

    def __init__(self, args: Any) -> None:
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

    def forward(
        self, graph: Union[Data, Batch], return_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with feature masking for the second expert.

        Args:
            graph:
                PyTorch Geometric data object with node features and graph structure
            return_weights:
                If True, also return the routing weights assigned to each expert

        Returns:
            y:
                Weighted combination of expert outputs, shape [batch_size, output_dim]
            weights (optional):
                Routing weights assigned to each expert, shape [batch_size, num_experts]
                Each row contains weights for one graph (sums to 1.0 due to softmax).
                Example: weights[0] = [0.7, 0.3] means graph 0 uses 70% expert 0, 30% expert 1.
                Only returned if return_weights=True
        """
        # Router computes logits for each graph in the batch
        logits = self.router(
            graph
        )  # [B, num_experts] - one logit per expert for each graph
        # Apply softmax to get normalized weights (each row sums to 1.0)
        weights = F.softmax(
            logits, dim=-1
        )  # [B, num_experts] - per-graph routing weights

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
        if return_weights:
            return y, weights
        return y
