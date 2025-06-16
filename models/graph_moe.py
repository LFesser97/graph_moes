import copy
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import global_mean_pool

from models.graph_model import GNN

class MoE(nn.Module):
    def __init__(self, args):
        """
        args should have attributes:
            args.layer_types:      list of two strings, e.g. ["GCN", "GIN"]
            args.input_dim:        int, node‐feature dim
            args.hidden_layers:    list of ints, one per hidden layer
            args.output_dim:       int, graph‐level output dim
            args.num_relations:    int, for relational GNNs
            args.mlp:              bool, whether to include final MLP layer
            args.dropout:          float, dropout probability
            args.last_layer_fa:    bool, whether to apply FA on last layer
        """
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