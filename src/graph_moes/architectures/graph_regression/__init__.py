"""Graph regression models."""

from graph_moes.architectures.graph_regression.graph_regression_model import GINE
from graph_moes.architectures.graph_regression.graph_regression_model import (
    GNN as RegressionGNN,
)
from graph_moes.architectures.graph_regression.graph_regression_model import (
    GPS as RegressionGPS,
)
from graph_moes.architectures.graph_regression.graph_regression_model import (
    OrthogonalGCN as RegressionOrthogonalGCN,
)
from graph_moes.architectures.graph_regression.graph_regression_model import (
    UnitaryGCN as RegressionUnitaryGCN,
)

__all__ = [
    "GINE",
    "RegressionGNN",
    "RegressionGPS",
    "RegressionOrthogonalGCN",
    "RegressionUnitaryGCN",
]
