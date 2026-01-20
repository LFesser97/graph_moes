import argparse
import ast

try:
    from attrdict3 import AttrDict  # Python 3.10+ compatible
except ImportError:
    from attrdict import AttrDict  # Fallback for older Python


def get_args_from_input() -> AttrDict:
    """
    Parse command-line arguments for network hyperparameters.

    Returns:
        AttrDict: Dictionary-like object containing parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="modify network parameters", argument_default=argparse.SUPPRESS
    )

    parser.add_argument("--learning_rate", metavar="", type=float, help="learning rate")
    parser.add_argument(
        "--max_epochs",
        metavar="",
        type=int,
        help="maximum number of epochs for training",
    )
    parser.add_argument(
        "--layer_type", metavar="", help="type of layer in GNN (GCN, GIN, GAT, etc.)"
    )
    parser.add_argument(
        "--display",
        metavar="",
        type=bool,
        help="toggle display messages showing training progress",
    )
    parser.add_argument(
        "--device", metavar="", type=str, help="name of CUDA device to use or CPU"
    )
    parser.add_argument(
        "--eval_every",
        metavar="X",
        type=int,
        help="calculate validation/test accuracy every X epochs",
    )
    parser.add_argument(
        "--stopping_criterion",
        metavar="",
        type=str,
        help="model stops training when this criterion stops improving (can be train, validation, or test)",
    )
    parser.add_argument(
        "--stopping_threshold",
        metavar="T",
        type=float,
        help="model perceives no improvement when it does worse than (best loss) * T",
    )
    parser.add_argument(
        "--patience",
        metavar="P",
        type=int,
        help="model stops training after P epochs with no improvement",
    )
    parser.add_argument(
        "--train_fraction",
        metavar="",
        type=float,
        help="fraction of the dataset to be used for training",
    )
    parser.add_argument(
        "--validation_fraction",
        metavar="",
        type=float,
        help="fraction of the dataset to be used for validation",
    )
    parser.add_argument(
        "--test_fraction",
        metavar="",
        type=float,
        help="fraction of the dataset to be used for testing",
    )
    parser.add_argument(
        "--dropout", metavar="", type=float, help="layer dropout probability"
    )
    parser.add_argument(
        "--weight_decay",
        metavar="",
        type=float,
        help="weight decay added to loss function",
    )
    parser.add_argument(
        "--hidden_dim", metavar="", type=int, help="width of hidden layer"
    )
    parser.add_argument(
        "--hidden_layers",
        metavar="",
        type=ast.literal_eval,
        help="list containing dimensions of all hidden layers",
    )
    parser.add_argument(
        "--num_layers", metavar="", type=int, help="number of hidden layers"
    )
    parser.add_argument(
        "--num_splits", metavar="", type=int, default=3, help="number of random splits"
    )
    parser.add_argument(
        "--batch_size",
        metavar="",
        type=int,
        help="number of samples in each training batch",
    )
    parser.add_argument(
        "--num_trials",
        metavar="",
        type=int,
        help="number of times the network is trained",
    )
    parser.add_argument(
        "--rewiring", metavar="", type=str, help="type of rewiring to be performed"
    )
    parser.add_argument(
        "--num_iterations",
        metavar="",
        type=int,
        help="number of iterations of rewiring",
    )
    parser.add_argument("--alpha", type=float, help="alpha hyperparameter for DIGL")
    parser.add_argument("--k", type=int, help="k hyperparameter for DIGL")
    parser.add_argument("--eps", type=float, help="epsilon hyperparameter for DIGL")
    parser.add_argument("--dataset", type=str, help="name of dataset to use")
    parser.add_argument(
        "--last_layer_fa",
        type=str,
        help="whether or not to make last layer fully adjacent",
    )
    parser.add_argument("--borf_batch_add", type=int, help="BORF batch addition size")
    parser.add_argument("--borf_batch_remove", type=int, help="BORF batch removal size")
    # DEPRECATED: On-the-fly encoding computation is deprecated in favor of pre-computed
    # dataset encodings via --dataset_encoding. This argument is kept for backwards
    # compatibility but should not be used in new experiments. Use --dataset_encoding instead.
    parser.add_argument(
        "--encoding",
        type=str,
        help="type of encoding to use for node features (DEPRECATED: use --dataset_encoding instead)",
    )
    parser.add_argument(
        "--dataset_encoding",
        type=str,
        default=None,
        help="pre-computed dataset encoding to use: None (normal), hg_ldp, hg_frc, hg_rwpe_we_k20, hg_lape_normalized_k8 (hypergraph), or g_ldp, g_rwpe_k16, g_lape_k8, g_orc (graph)",
    )
    parser.add_argument(
        "--encoding_moe_encodings",
        type=str,
        nargs="+",
        default=None,
        help="list of encodings for EncodingMoE (e.g., --encoding_moe_encodings g_ldp g_orc). If specified, enables EncodingMoE model that routes between encodings.",
    )
    parser.add_argument(
        "--encoding_moe_router_type",
        type=str,
        default="MLP",
        choices=["MLP", "GNN"],
        help="router type for EncodingMoE: MLP or GNN (default: MLP)",
    )
    parser.add_argument(
        "--router_hidden_layers",
        type=list,
        default=[64, 64, 64],
        help="num. hidden layers of the router GNN",
    )
    parser.add_argument(
        "--layer_types",
        type=ast.literal_eval,
        help="the expert GNNs to be used for an MoE",
    )
    parser.add_argument(
        "--skip_connection",
        action="store_true",
        default=False,
        help="whether to use skip/residual connections (for GCN, GIN, SAGE)",
    )
    parser.add_argument(
        "--normalize_features",
        action="store_true",
        default=False,
        help="whether to normalize node features (L2 normalization per node)",
    )

    # WandB arguments
    parser.add_argument(
        "--wandb_enabled",
        action="store_true",
        default=False,
        help="enable wandb logging",
    )
    parser.add_argument(
        "--wandb_project", type=str, default="MOE_4", help="wandb project name"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="weber-geoml-harvard-university",
        help="wandb entity/team name",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="wandb run name (auto-generated if not provided)",
    )
    parser.add_argument(
        "--wandb_dir", type=str, default="./wandb", help="directory to save wandb logs"
    )
    parser.add_argument(
        "--wandb_tags",
        type=ast.literal_eval,
        default=None,
        help="list of tags for wandb run",
    )

    args = parser.parse_args()
    return AttrDict(vars(args))
