"""Measures the Dirichlet energy of a vector field on a graph."""

import numpy as np
from numba import jit


@jit(nopython=True)
def dirichlet_energy(X: np.ndarray, edge_index: np.ndarray) -> float:
    """Compute the Dirichlet energy of a vector field X on a graph.

    The Dirichlet energy measures the smoothness of a vector field over a graph.
    It is computed as the sum of squared differences between connected nodes,
    normalized by their degrees.

    Args:
        X:
            Node feature matrix of shape (n_nodes, n_features) containing
            the vector field values at each node.
        edge_index:
            Edge connectivity matrix of shape (2, n_edges) where
            edge_index[0] contains source nodes and edge_index[1] contains
            target nodes.

    Returns:
        The Dirichlet energy as a scalar value. Higher values indicate
        less smooth vector fields.

    Note:
        This function is JIT-compiled with numba for performance.
    """
    # computes Dirichlet energy of a vector field X with respect to a graph with a given edge index
    n = X.shape[0]
    m = len(edge_index[0])
    num_features = X.shape[1]
    degrees = np.zeros(n)
    for edge_idx in range(m):
        u = edge_index[0][edge_idx]
        degrees[u] += 1
    y = np.linalg.norm(X.flatten()) ** 2
    for edge_idx in range(m):
        for i in range(num_features):
            u = edge_index[0][edge_idx]
            v = edge_index[1][edge_idx]
            y -= X[u][i] * X[v][i] / (degrees[u] * degrees[v]) ** 0.5
    return y


def dirichlet_normalized(X: np.ndarray, edge_index: np.ndarray) -> float:
    """Compute the normalized Dirichlet energy of a vector field X on a graph.

    This function computes the Dirichlet energy normalized by the squared
    Frobenius norm of the vector field, providing a scale-invariant measure
    of smoothness.

    Args:
        X:
            Node feature matrix of shape (n_nodes, n_features) containing
            the vector field values at each node.
        edge_index: Edge connectivity matrix of shape (2, n_edges) where
            edge_index[0] contains source nodes and edge_index[1] contains
            target nodes.

    Returns:
        The normalized Dirichlet energy as a scalar value between 0 and 1.
        Values closer to 0 indicate smoother vector fields.
    """
    energy = dirichlet_energy(X, edge_index)
    norm_squared = sum(sum(X**2))
    return energy / norm_squared
