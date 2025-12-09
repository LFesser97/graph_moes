"""Graph neural network architectures."""

from graph_moes.architectures.graph_model import (  # noqa: F401
    GNN,
    GPS,
    OrthogonalGCN,
    ResGatedGraphConv,
    RGATConv,
    RGINConv,
    UnitaryGCN,
)

__all__ = [
    "GNN",
    "GPS",
    "UnitaryGCN",
    "OrthogonalGCN",
    "ResGatedGraphConv",
    "RGATConv",
    "RGINConv",
]
