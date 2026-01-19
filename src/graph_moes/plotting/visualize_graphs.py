"""
Utilities for visualizing PyTorch Geometric graph datasets.

This module provides functions to convert PyG graphs to NetworkX format
and visualize them using matplotlib.
"""

from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx  # type: ignore
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


def visualize_graph(
    graph: Data,
    ax: Optional[plt.Axes] = None,
    node_color: Optional[str] = None,
    node_size: int = 300,
    edge_width: float = 1.0,
    title: Optional[str] = None,
    pos: Optional[nx.spring_layout] = None,
    with_labels: bool = False,
    cmap: Optional[str] = None,
) -> plt.Axes:
    """
    Visualize a single PyTorch Geometric graph.

    Args:
        graph: PyG Data object to visualize
        ax: Matplotlib axes to plot on. If None, creates a new figure
        node_color: Color for nodes. Can be a string, or 'y' to use labels if available
        node_size: Size of nodes
        edge_width: Width of edges
        title: Title for the plot
        pos: Node positions (layout). If None, uses spring layout
        with_labels: Whether to show node labels
        cmap: Colormap for node colors

    Returns:
        Matplotlib axes object
    """
    # Convert PyG graph to NetworkX
    import networkx as nx

    # Handle graphs with no edges or None edge_index
    if not hasattr(graph, "edge_index") or graph.edge_index is None:
        # Graph with no edges - create node-only graph
        G = nx.Graph()
        num_nodes = None

        # Try to get number of nodes from various sources
        if hasattr(graph, "x") and graph.x is not None:
            try:
                if hasattr(graph.x, "shape") and len(graph.x.shape) > 0:
                    num_nodes = graph.x.shape[0]
                elif hasattr(graph.x, "__len__"):
                    num_nodes = len(graph.x)
            except (AttributeError, TypeError, IndexError):
                pass

        if (
            num_nodes is None
            and hasattr(graph, "num_nodes")
            and graph.num_nodes is not None
        ):
            try:
                num_nodes = int(graph.num_nodes)
            except (ValueError, TypeError):
                pass

        # If we still don't have num_nodes, try to infer from other attributes
        if num_nodes is None:
            # Check if there are other node-related attributes
            for attr_name in ["node_attr", "node_feat", "features"]:
                if hasattr(graph, attr_name):
                    attr = getattr(graph, attr_name)
                    if attr is not None:
                        try:
                            if hasattr(attr, "shape") and len(attr.shape) > 0:
                                num_nodes = attr.shape[0]
                                break
                            elif hasattr(attr, "__len__"):
                                num_nodes = len(attr)
                                break
                        except (AttributeError, TypeError, IndexError):
                            pass

        # If still None, check node_stores (for HeteroData or special PyG formats)
        if num_nodes is None and hasattr(graph, "node_stores"):
            try:
                if hasattr(graph.node_stores, "__len__") and len(graph.node_stores) > 0:
                    ns = graph.node_stores[0]
                    # Try dictionary-like access first
                    if hasattr(ns, "get") or hasattr(ns, "__contains__"):
                        try:
                            if "x" in ns:
                                x_val = ns["x"]
                                if (
                                    x_val is not None
                                    and hasattr(x_val, "shape")
                                    and len(x_val.shape) > 0
                                ):
                                    num_nodes = x_val.shape[0]
                            elif "num_nodes" in ns:
                                num_nodes = ns["num_nodes"]
                        except (KeyError, TypeError):
                            pass

                    # Try attribute-like access
                    if num_nodes is None:
                        if hasattr(ns, "x") and ns.x is not None:
                            if hasattr(ns.x, "shape") and len(ns.x.shape) > 0:
                                num_nodes = ns.x.shape[0]
                        elif hasattr(ns, "num_nodes") and ns.num_nodes is not None:
                            num_nodes = ns.num_nodes

                    # Last resort: check all attributes of node_store
                    if num_nodes is None:
                        for attr in ["x", "num_nodes", "node_feat", "features"]:
                            if hasattr(ns, attr):
                                val = getattr(ns, attr)
                                if val is not None:
                                    if hasattr(val, "shape") and len(val.shape) > 0:
                                        num_nodes = val.shape[0]
                                        break
                                    elif hasattr(val, "__len__"):
                                        num_nodes = len(val)
                                        break
            except (AttributeError, TypeError, IndexError, KeyError):
                pass

        # Add nodes to graph
        if num_nodes is not None and num_nodes > 0:
            G.add_nodes_from(range(num_nodes))
        else:
            # Last resort: try to get from graph's internal structure
            # Some graphs might have the data but it's not accessible via standard attributes
            try:
                # Check if graph has a way to compute num_nodes
                if hasattr(graph, "__len__"):
                    # Some graph types use __len__ for num_nodes
                    try:
                        num_nodes = len(graph)
                        if num_nodes > 0:
                            G.add_nodes_from(range(num_nodes))
                    except (TypeError, ValueError):
                        pass
            except Exception:
                pass

            # Final fallback: at least show one node (but this might be correct if graph truly has 1 node)
            if len(G) == 0:
                G.add_node(0)
    elif graph.edge_index.numel() == 0:
        # Empty edge_index - create node-only graph
        G = nx.Graph()
        if hasattr(graph, "x") and graph.x is not None:
            try:
                num_nodes = (
                    graph.x.shape[0]
                    if hasattr(graph.x, "shape")
                    else len(graph.x) if hasattr(graph.x, "__len__") else 1
                )
                G.add_nodes_from(range(num_nodes))
            except (AttributeError, TypeError):
                G.add_node(0)
        elif hasattr(graph, "num_nodes") and graph.num_nodes is not None:
            G.add_nodes_from(range(graph.num_nodes))
        else:
            G.add_node(0)
    else:
        try:
            G = to_networkx(graph, to_undirected=True, remove_self_loops=True)
        except Exception:
            # Fallback: create graph manually
            G = nx.Graph()
            try:
                edge_index = graph.edge_index.cpu().numpy()
                if edge_index.shape[0] == 2:
                    edges = list(zip(edge_index[0], edge_index[1]))
                    G.add_edges_from(edges)
            except Exception:
                pass
            # Add nodes if they exist
            if hasattr(graph, "x") and graph.x is not None:
                try:
                    num_nodes = (
                        graph.x.shape[0]
                        if hasattr(graph.x, "shape")
                        else len(graph.x) if hasattr(graph.x, "__len__") else 1
                    )
                    for i in range(num_nodes):
                        if i not in G:
                            G.add_node(i)
                except (AttributeError, TypeError):
                    pass

    # Create figure if no axes provided
    if ax is None:
        _fig, ax = plt.subplots(figsize=(8, 8))

    # Determine node colors
    # Note: graph.y is graph-level (single value), not node-level, so we can't use it for node colors
    if isinstance(node_color, str) and node_color != "y":
        # Use provided color string
        node_colors: Union[str, np.ndarray, List[float]] = node_color
    elif node_color == "y" and hasattr(graph, "x") and graph.x is not None:
        # Use node features for coloring (first feature dimension)
        node_features = graph.x.detach().cpu().numpy()
        if node_features.ndim > 1 and node_features.shape[0] == len(G.nodes):
            # Use first feature dimension for coloring
            node_colors = node_features[:, 0]
        elif node_features.ndim == 1 and len(node_features) == len(G.nodes):
            node_colors = node_features
        else:
            # Fallback to degree-based coloring
            degrees = dict(G.degree())
            node_colors = [degrees[n] for n in G.nodes()]
    else:
        # Default: color by degree
        degrees = dict(G.degree())
        node_colors = [degrees[n] for n in G.nodes()]

    # Generate layout if not provided - use spring layout with better parameters
    if pos is None:
        # Adjust k based on graph size for better spacing
        num_nodes = len(G.nodes())
        k_param = (
            max(1.0, min(3.0, np.sqrt(1.0 / num_nodes) * 10)) if num_nodes > 0 else 1.0
        )
        pos = nx.spring_layout(G, seed=42, k=k_param, iterations=100)

    # Draw edges
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        alpha=0.5,
        width=edge_width,
        edge_color="gray",
    )

    # Draw nodes
    if isinstance(node_colors, str):
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_color=node_colors,
            node_size=node_size,
            alpha=0.9,
        )
    else:
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_color=node_colors,
            node_size=node_size,
            alpha=0.9,
            cmap=cmap if cmap is not None else "viridis",
        )

    # Draw labels if requested
    if with_labels:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)

    # Set title
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")

    # Get graph statistics for subtitle
    num_nodes = (
        graph.num_nodes
        if hasattr(graph, "num_nodes") and graph.num_nodes is not None
        else 0
    )
    if (
        hasattr(graph, "edge_index")
        and graph.edge_index is not None
        and graph.edge_index.shape[0] > 0
    ):
        num_edges = (
            graph.edge_index.shape[1] // 2
            if len(graph.edge_index.shape) > 1 and graph.edge_index.shape[1] > 0
            else 0
        )
    else:
        num_edges = 0
    stats_text = f"Nodes: {num_nodes}, Edges: {num_edges}"

    # Add label information if available
    if hasattr(graph, "y") and graph.y is not None:
        try:
            label = graph.y.item() if graph.y.numel() == 1 else graph.y
            stats_text += f", Label: {label}"
        except (ValueError, RuntimeError, AttributeError):
            # If label can't be converted to item, skip it
            pass

    ax.text(
        0.5,
        -0.05,
        stats_text,
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        style="italic",
    )

    ax.axis("off")

    return ax


def visualize_graph_grid(
    graphs: List[Data],
    n_cols: int = 4,
    n_rows: Optional[int] = None,
    figsize: Tuple[int, int] = (16, 16),
    titles: Optional[List[str]] = None,
    dataset_name: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize multiple graphs in a grid layout.

    Args:
        graphs: List of PyG Data objects to visualize
        n_cols: Number of columns in the grid
        n_rows: Number of rows in the grid. If None, calculated automatically
        figsize: Figure size (width, height)
        titles: List of titles for each graph. If None, uses indices
        dataset_name: Name of the dataset for the overall figure title

    Returns:
        Matplotlib figure object
    """
    n_graphs = len(graphs)
    if n_rows is None:
        n_rows = (n_graphs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Handle single row/column case
    if n_rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    axes = axes.flatten()

    for i in range(n_rows * n_cols):
        ax = axes[i]
        if i < n_graphs:
            title = titles[i] if titles and i < len(titles) else f"Graph {i+1}"
            # Use degree-based coloring instead of "y" since y is graph-level
            # Use None for node_color to default to degree-based coloring
            visualize_graph(graphs[i], ax=ax, title=title, node_color=None)
        else:
            ax.axis("off")

    # Set overall title
    if dataset_name:
        fig.suptitle(
            f"Example Graphs from {dataset_name}",
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )

    plt.tight_layout(rect=(0, 0, 1, 0.99))
    return fig


def visualize_single_graph(
    graph: Data,
    figsize: Tuple[int, int] = (10, 10),
    node_size: int = 500,
    edge_width: float = 1.5,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a standalone visualization of a single graph with spring layout.

    Args:
        graph: PyG Data object to visualize
        figsize: Figure size (width, height)
        node_size: Size of nodes
        edge_width: Width of edges
        title: Title for the plot
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    visualize_graph(
        graph,
        ax=ax,
        node_color=None,
        node_size=node_size,
        edge_width=edge_width,
        title=title,
        with_labels=False,
    )
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def sample_graphs(
    graphs: List[Data], n_samples: int = 8, random_seed: Optional[int] = None
) -> List[Data]:
    """
    Sample a random subset of graphs from a list.

    Args:
        graphs: List of graphs to sample from
        n_samples: Number of graphs to sample
        random_seed: Random seed for reproducibility

    Returns:
        List of sampled graphs
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if len(graphs) <= n_samples:
        return graphs

    indices = np.random.choice(len(graphs), size=n_samples, replace=False)
    return [graphs[i] for i in sorted(indices)]
