"""EncodingMixture: Dynamic encoding selection using routing mechanism."""

from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch, Data

from graph_moes.routing_encodings.encoding_router import EncodingRouter


class EncodingMoE(nn.Module):
    """Encoding Mixture of Experts: Dynamic encoding selection based on graph characteristics.

    This model uses a router (MLP or GNN) to dynamically select which encoding to append
    to the base node features for each graph. Different graphs in the dataset can receive
    different encodings based on their structural characteristics.

    The model:
    1. Starts with base features (original node features without encoding)
    2. Uses a router to compute weights for different encoding options
    3. Selects one encoding (or weighted combination) to append
    4. Concatenates base features + selected encoding
    5. Processes through a GNN model as usual

    Args:
        args: Configuration object with:
            - base_input_dim: Dimension of base features (before encoding)
            - encoding_configs: List of encoding configurations, each with:
                - encoding_name: str (e.g., "hg_lape_normalized_k8", "g_rwpe_k16")
                - encoding_dim: int (dimension of encoding features to append)
                - replaces_base: bool (True if encoding replaces base, False if appends)
            - router_type: str ("MLP" or "GNN")
            - gnn_model: The GNN model to use after encoding selection
            - Other router-specific parameters
    """

    def __init__(
        self, args: Any, encoding_configs: List[Dict[str, Any]], gnn_model: nn.Module
    ) -> None:
        """Initialize EncodingMoE.

        Args:
            args: Configuration object
            encoding_configs: List of encoding configuration dicts, each with:
                - encoding_name: str
                - encoding_dim: int (dimension of encoding features)
                - replaces_base: bool (whether encoding replaces base features)
            gnn_model: The GNN model to use after encoding selection
        """
        super().__init__()
        self.args = args
        self.encoding_configs = encoding_configs
        self.gnn_model = gnn_model
        self.num_encodings = len(encoding_configs)

        # Store normalization flag
        self.normalize_features = getattr(args, "normalize_features", False)

        # Base input dimension (before encoding)
        self.base_input_dim = args.base_input_dim

        # Initialize router (works on base features)
        router_args = type(
            "Args",
            (),
            {
                "router_hidden_layers": getattr(
                    args, "router_hidden_layers", [64, 64, 64]
                ),
                "router_layer_type": getattr(args, "router_layer_type", "GIN"),
                "router_depth": getattr(args, "router_depth", 4),
                "router_dropout": getattr(args, "router_dropout", 0.1),
            },
        )()

        self.router = EncodingRouter(
            router_type=args.router_type,
            input_dim=self.base_input_dim,
            num_encodings=self.num_encodings,
            args=router_args,
        )

        # Store encoding dimensions for dimension checking
        self.encoding_dims = [config["encoding_dim"] for config in encoding_configs]

    def extract_encoding_features(
        self, base_graph: Data, encoded_graph: Data, encoding_config: Dict[str, Any]
    ) -> torch.Tensor:
        """Extract just the encoding features from an encoded graph.

        Handles two cases:
        1. Encoding appends to base (replaces_base=False): extract dims after base_input_dim
        2. Encoding replaces base (replaces_base=True): use all features as encoding

        Args:
            base_graph: Graph with base features (original, no encoding)
            encoded_graph: Graph with precomputed encoding (may include base + encoding)
            encoding_config: Configuration dict with 'encoding_dim' and 'replaces_base'

        Returns:
            encoding_features: Tensor of shape [num_nodes, encoding_dim] with just the encoding
        """
        if encoding_config.get("replaces_base", False):
            # Encoding replaces base features - use all features
            result = torch.as_tensor(encoded_graph.x, dtype=torch.float32)
            return result
        else:
            # Encoding appends to base - extract the encoding part
            base_dim = base_graph.x.shape[1]
            encoded_dim = encoded_graph.x.shape[1]
            encoding_dim = encoding_config["encoding_dim"]

            # Check if encoded graph has base + encoding
            if encoded_dim >= base_dim + encoding_dim:
                # Extract just the encoding part (after base dims)
                result = torch.as_tensor(
                    encoded_graph.x[:, base_dim : base_dim + encoding_dim],
                    dtype=torch.float32,
                )
                return result
            elif encoded_dim == encoding_dim:
                # Encoded graph has just encoding (unlikely but handle it)
                result = torch.as_tensor(encoded_graph.x, dtype=torch.float32)
                return result
            else:
                # Fallback: assume last encoding_dim dims are encoding
                result = torch.as_tensor(
                    encoded_graph.x[:, -encoding_dim:], dtype=torch.float32
                )
                return result

    def forward(
        self,
        base_graph: Union[Data, Batch],
        encoded_graphs: Dict[str, Union[Data, Batch]],
        return_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through EncodingMoE.

        Works like regular MoE: each encoding is processed separately through the GNN,
        then router weights are used to combine the outputs.

        Process:
        1. Router computes weights based on base features
        2. For each encoding: (base + encoding) → GNN → output_i
        3. Weighted combination: final_output = sum(weights[i] * output_i)

        Args:
            base_graph: Graph with base features (original, no encoding)
            encoded_graphs: Dict mapping encoding_name to graph with that encoding
            return_weights: If True, also return routing weights

        Returns:
            output: Graph-level output from weighted combination of expert outputs
            weights (optional): Routing weights shape [batch_size, num_encodings]
        """
        # Normalize base features if requested (before routing)
        if self.normalize_features:
            x_norm = torch.norm(base_graph.x.float(), p=2, dim=1, keepdim=True)
            base_graph.x = base_graph.x / (x_norm + 1e-8)

        # Router computes logits for each graph in batch based on base features
        logits = self.router(base_graph)  # [batch_size, num_encodings]
        weights = F.softmax(logits, dim=-1)  # [batch_size, num_encodings]

        # Process each encoding separately through GNN (like experts in regular MoE)
        expert_outputs = []
        for config in self.encoding_configs:
            encoding_name = config["encoding_name"]

            # Extract encoding features
            if encoding_name not in encoded_graphs:
                # Missing encoding - use zero encoding as fallback
                num_nodes = base_graph.x.shape[0]
                encoding_dim = config["encoding_dim"]
                encoding_features = torch.zeros(
                    (num_nodes, encoding_dim),
                    dtype=base_graph.x.dtype,
                    device=base_graph.x.device,
                )
            else:
                encoded_graph = encoded_graphs[encoding_name]
                encoding_features = self.extract_encoding_features(
                    base_graph, encoded_graph, config
                )

            # Create augmented graph: base + this encoding
            augmented_graph = base_graph.clone()
            augmented_features = torch.cat(
                [base_graph.x.float(), encoding_features], dim=-1
            )
            augmented_graph.x = augmented_features

            # Process through GNN (this is one "expert")
            expert_output = self.gnn_model(augmented_graph)
            expert_outputs.append(expert_output)

        # Weighted combination of expert outputs (like regular MoE)
        # weights: [batch_size, num_encodings]
        # expert_outputs: list of [batch_size, output_dim]
        # Final output: weighted sum
        output = sum(
            weights[:, i].unsqueeze(-1) * expert_outputs[i]
            for i in range(len(expert_outputs))
        )  # [batch_size, output_dim]

        if return_weights:
            return output, weights
        return output
