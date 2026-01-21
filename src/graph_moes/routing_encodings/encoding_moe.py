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

        # Store expected input dimension from GNN model (first layer input size)
        # This is the input_dim the GNN was initialized with (max across all encodings)
        if hasattr(gnn_model, "layers") and len(gnn_model.layers) > 0:
            first_layer = gnn_model.layers[0]
            if hasattr(first_layer, "lin"):
                # Get from first layer weight matrix (weight shape is [out_dim, in_dim])
                self.expected_input_dim = first_layer.lin.weight.shape[1]
            elif hasattr(first_layer, "weight"):
                # Some layers might have weight directly
                self.expected_input_dim = (
                    first_layer.weight.shape[1]
                    if len(first_layer.weight.shape) > 1
                    else args.base_input_dim + max(self.encoding_dims)
                )
            else:
                # Fallback: calculate from args
                self.expected_input_dim = args.base_input_dim + max(self.encoding_dims)
        elif hasattr(gnn_model, "args"):
            # Get from GNN model args
            self.expected_input_dim = gnn_model.args.input_dim
        else:
            # Fallback: calculate from args (max across all encoding combinations)
            max_input_dims = []
            for config in encoding_configs:
                if config.get("replaces_base", False):
                    max_input_dims.append(config["encoding_dim"])
                else:
                    max_input_dims.append(args.base_input_dim + config["encoding_dim"])
            self.expected_input_dim = (
                max(max_input_dims) if max_input_dims else args.base_input_dim
            )

    def extract_encoding_features(
        self, base_graph: Data, encoded_graph: Data, encoding_config: Dict[str, Any]
    ) -> torch.Tensor:
        """Extract just the encoding features from an encoded graph.

        NOTE: On the cluster, encoding files are saved as [original features] + [encodings].
        So encoded_graph.x has shape [num_nodes, base_dim + encoding_dim].

        Handles two cases:
        1. Encoding appends to base (replaces_base=False): extract dims after base_input_dim
        2. Encoding replaces base (replaces_base=True): extract last encoding_dim dims (the encoding part)

        Args:
            base_graph: Graph with base features (original, no encoding)
            encoded_graph: Graph with precomputed encoding (saved as base + encoding)
            encoding_config: Configuration dict with 'encoding_dim' and 'replaces_base'

        Returns:
            encoding_features: Tensor of shape [num_nodes, encoding_dim] with just the encoding
        """
        base_dim = base_graph.x.shape[1]
        encoded_dim = encoded_graph.x.shape[1]
        encoding_dim = encoding_config["encoding_dim"]

        if encoding_config.get("replaces_base", False):
            # Encoding replaces base - extract the encoding part (last encoding_dim dims)
            # Even though file has base + encoding, we only use the encoding part
            if encoded_dim >= base_dim + encoding_dim:
                # File has base + encoding: extract just the encoding part (after base)
                result = torch.as_tensor(
                    encoded_graph.x[:, base_dim : base_dim + encoding_dim],
                    dtype=torch.float32,
                )
                return result
            elif encoded_dim == encoding_dim:
                # File has just encoding (unlikely but handle it)
                result = torch.as_tensor(encoded_graph.x, dtype=torch.float32)
                return result
            else:
                # Fallback: assume last encoding_dim dims are encoding
                result = torch.as_tensor(
                    encoded_graph.x[:, -encoding_dim:], dtype=torch.float32
                )
                return result
        else:
            # Encoding appends to base - extract the encoding part (after base dims)
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

            # Create augmented graph: base + this encoding OR just encoding if replaces_base
            augmented_graph = base_graph.clone()
            if config.get("replaces_base", False):
                # Encoding replaces base - use only encoding features (no base)
                augmented_features = encoding_features.float()
            else:
                # Encoding appends to base - concatenate base + encoding
                augmented_features = torch.cat(
                    [base_graph.x.float(), encoding_features.float()], dim=-1
                )

            # Ensure features match the expected input dimension (pad if necessary)
            # The GNN model was initialized with expert_input_dim (max across all encodings)
            # So we need to pad/align all inputs to this dimension
            if augmented_features.shape[1] < self.expected_input_dim:
                # Pad with zeros to match expected dimension
                num_nodes = augmented_features.shape[0]
                pad_size = self.expected_input_dim - augmented_features.shape[1]
                padding = torch.zeros(
                    (num_nodes, pad_size),
                    dtype=augmented_features.dtype,
                    device=augmented_features.device,
                )
                augmented_features = torch.cat([augmented_features, padding], dim=-1)
            elif augmented_features.shape[1] > self.expected_input_dim:
                # Truncate if somehow larger (shouldn't happen but handle gracefully)
                augmented_features = augmented_features[:, : self.expected_input_dim]

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
