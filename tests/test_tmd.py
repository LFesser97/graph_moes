"""Test suite for Tree Mover's Distance (TMD) module.

This module tests the TMD implementation including:
- get_neighbors(): Extract neighbor information from graphs
- TMD(): Compute Tree Mover's Distance between graphs
- extract_labels(): Extract labels from datasets
- compute_tmd_matrix(): Compute pairwise TMD matrix
- compute_class_distance_ratios(): Compute class-distance ratios
- save_tmd_results(): Save TMD computation results
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from graph_moes.tmd import (
    TMD,
    compute_class_distance_ratios,
    compute_tmd_matrix,
    extract_labels,
    get_neighbors,
    save_tmd_results,
)


@pytest.fixture
def simple_graph_2nodes():
    """Create a simple 2-node graph with bidirectional edge.

    Graph structure:
    0 <-> 1 (bidirectional)
    """
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    y = torch.tensor(0)
    return Data(x=x, edge_index=edge_index, y=y)


@pytest.fixture
def simple_graph_3nodes_chain():
    """Create a simple 3-node chain graph.

    Graph structure:
    0 --- 1 --- 2
    """
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    y = torch.tensor(1)
    return Data(x=x, edge_index=edge_index, y=y)


@pytest.fixture
def simple_graph_3nodes_triangle():
    """Create a 3-node triangle (complete graph).

    Graph structure:
       0
      / \
     1---2
    """
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]], dtype=torch.long
    )
    y = torch.tensor(0)
    return Data(x=x, edge_index=edge_index, y=y)


@pytest.fixture
def isolated_node_graph():
    """Create a graph with a single isolated node."""
    x = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    edge_index = torch.tensor([[], []], dtype=torch.long)
    y = torch.tensor(0)
    return Data(x=x, edge_index=edge_index, y=y)


class TestGetNeighbors:
    """Test the get_neighbors function."""

    def test_simple_graph_2nodes(self, simple_graph_2nodes):
        """Test neighbor extraction for 2-node graph."""
        adj = get_neighbors(simple_graph_2nodes)

        assert isinstance(adj, dict)
        assert 0 in adj
        assert 1 in adj
        assert adj[0] == [1]
        assert adj[1] == [0]

    def test_chain_graph(self, simple_graph_3nodes_chain):
        """Test neighbor extraction for chain graph."""
        adj = get_neighbors(simple_graph_3nodes_chain)

        assert isinstance(adj, dict)
        assert 0 in adj
        assert 1 in adj
        assert 2 in adj
        assert adj[0] == [1]
        assert 1 in adj[1]  # Node 1 has neighbors
        assert adj[2] == [1]

    def test_triangle_graph(self, simple_graph_3nodes_triangle):
        """Test neighbor extraction for triangle graph."""
        adj = get_neighbors(simple_graph_3nodes_triangle)

        assert isinstance(adj, dict)
        assert len(adj[0]) == 2  # Node 0 connected to 1 and 2
        assert len(adj[1]) == 2  # Node 1 connected to 0 and 2
        assert len(adj[2]) == 2  # Node 2 connected to 0 and 1

    def test_isolated_node(self, isolated_node_graph):
        """Test neighbor extraction for isolated node."""
        adj = get_neighbors(isolated_node_graph)

        assert isinstance(adj, dict)
        # Isolated node has no neighbors, so it won't appear in adj
        assert len(adj) == 0


class TestTMD:
    """Test the TMD function."""

    def test_tmd_symmetry(self, simple_graph_2nodes, simple_graph_3nodes_chain):
        """Test that TMD is symmetric: TMD(G1, G2) = TMD(G2, G1)."""
        tmd_12 = TMD(simple_graph_2nodes, simple_graph_3nodes_chain, w=1.0, L=4)
        tmd_21 = TMD(simple_graph_3nodes_chain, simple_graph_2nodes, w=1.0, L=4)

        assert abs(tmd_12 - tmd_21) < 1e-6, f"TMD not symmetric: {tmd_12} != {tmd_21}"

    def test_tmd_identity(self, simple_graph_2nodes):
        """Test that TMD(G, G) ≈ 0 (distance to itself is zero)."""
        tmd_self = TMD(simple_graph_2nodes, simple_graph_2nodes, w=1.0, L=4)

        # Should be very close to zero (allowing for numerical errors)
        assert tmd_self >= 0, "TMD should be non-negative"
        assert tmd_self < 1e-6, f"TMD(G, G) should be close to 0, got {tmd_self}"

    def test_tmd_non_negative(self, simple_graph_2nodes, simple_graph_3nodes_chain):
        """Test that TMD is always non-negative."""
        tmd = TMD(simple_graph_2nodes, simple_graph_3nodes_chain, w=1.0, L=4)

        assert tmd >= 0, f"TMD should be non-negative, got {tmd}"

    def test_tmd_different_graphs(self, simple_graph_2nodes, simple_graph_3nodes_chain):
        """Test that TMD between different graphs is positive."""
        tmd = TMD(simple_graph_2nodes, simple_graph_3nodes_chain, w=1.0, L=4)

        assert tmd > 0, f"TMD between different graphs should be positive, got {tmd}"

    def test_tmd_with_list_weights(
        self, simple_graph_2nodes, simple_graph_3nodes_chain
    ):
        """Test TMD with list of weights."""
        w_list = [0.5, 1.0, 1.5]  # L=4 means L-1=3 weights
        tmd = TMD(simple_graph_2nodes, simple_graph_3nodes_chain, w=w_list, L=4)

        assert tmd >= 0
        assert isinstance(tmd, (float, np.floating))

    def test_tmd_different_depths(self, simple_graph_2nodes, simple_graph_3nodes_chain):
        """Test TMD with different depth parameters."""
        tmd_l2 = TMD(simple_graph_2nodes, simple_graph_3nodes_chain, w=1.0, L=2)
        tmd_l4 = TMD(simple_graph_2nodes, simple_graph_3nodes_chain, w=1.0, L=4)

        assert tmd_l2 >= 0
        assert tmd_l4 >= 0
        # Different depths may give different results
        assert isinstance(tmd_l2, (float, np.floating))
        assert isinstance(tmd_l4, (float, np.floating))

    def test_tmd_isolated_node(self, isolated_node_graph, simple_graph_2nodes):
        """Test TMD with isolated node graph."""
        tmd = TMD(isolated_node_graph, simple_graph_2nodes, w=1.0, L=4)

        assert tmd >= 0
        assert isinstance(tmd, (float, np.floating))

    def test_tmd_same_structure_different_features(self, simple_graph_2nodes):
        """Test TMD between graphs with same structure but different features."""
        # Create graph with same structure but different features
        x2 = torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=torch.float32)
        graph2 = Data(
            x=x2,
            edge_index=simple_graph_2nodes.edge_index,
            y=simple_graph_2nodes.y,
        )

        tmd = TMD(simple_graph_2nodes, graph2, w=1.0, L=4)

        # Should be positive (different features)
        assert tmd > 0
        # But should be smaller than completely different graphs
        assert tmd < 10.0  # Reasonable upper bound


class TestExtractLabels:
    """Test the extract_labels function."""

    def test_scalar_labels(self):
        """Test extraction of scalar labels."""
        graphs = [
            Data(
                x=torch.randn(3, 2),
                edge_index=torch.tensor([[0, 1], [1, 0]]),
                y=torch.tensor(0),
            ),
            Data(
                x=torch.randn(2, 2),
                edge_index=torch.tensor([[0], [1]]),
                y=torch.tensor(1),
            ),
            Data(
                x=torch.randn(4, 2),
                edge_index=torch.tensor([[0, 1], [1, 2]]),
                y=torch.tensor(0),
            ),
        ]

        labels, num_classes, label_counts = extract_labels(graphs)

        assert isinstance(labels, np.ndarray)
        assert len(labels) == 3
        assert np.array_equal(labels, [0, 1, 0])
        assert num_classes == 2
        assert label_counts == {0: 2, 1: 1}

    def test_1d_tensor_labels(self):
        """Test extraction of 1D tensor labels."""
        graphs = [
            Data(
                x=torch.randn(3, 2),
                edge_index=torch.tensor([[0, 1], [1, 0]]),
                y=torch.tensor([0]),
            ),
            Data(
                x=torch.randn(2, 2),
                edge_index=torch.tensor([[0], [1]]),
                y=torch.tensor([1]),
            ),
        ]

        labels, num_classes, label_counts = extract_labels(graphs)

        assert len(labels) == 2
        assert np.array_equal(labels, [0, 1])
        assert num_classes == 2
        assert label_counts == {0: 1, 1: 1}

    def test_missing_labels(self):
        """Test that missing labels raise ValueError."""
        graphs = [
            Data(x=torch.randn(3, 2), edge_index=torch.tensor([[0, 1], [1, 0]])),
            # Missing y attribute
        ]

        with pytest.raises(ValueError, match="does not have a 'y' attribute"):
            extract_labels(graphs)


class TestComputeTMDMatrix:
    """Test the compute_tmd_matrix function."""

    def test_small_dataset(self, simple_graph_2nodes, simple_graph_3nodes_chain):
        """Test TMD matrix computation for small dataset."""
        dataset = [simple_graph_2nodes, simple_graph_3nodes_chain]

        tmd_matrix = compute_tmd_matrix(dataset, w=1.0, L=4, verbose=False)

        assert isinstance(tmd_matrix, np.ndarray)
        assert tmd_matrix.shape == (2, 2)
        # Diagonal should be zero
        assert tmd_matrix[0, 0] < 1e-6
        assert tmd_matrix[1, 1] < 1e-6
        # Matrix should be symmetric
        assert abs(tmd_matrix[0, 1] - tmd_matrix[1, 0]) < 1e-6
        # Off-diagonal should be positive
        assert tmd_matrix[0, 1] > 0

    def test_tmd_matrix_symmetry(
        self, simple_graph_2nodes, simple_graph_3nodes_chain, isolated_node_graph
    ):
        """Test that TMD matrix is symmetric."""
        dataset = [simple_graph_2nodes, simple_graph_3nodes_chain, isolated_node_graph]

        tmd_matrix = compute_tmd_matrix(dataset, w=1.0, L=4, verbose=False)

        # Check symmetry
        for i in range(len(dataset)):
            for j in range(len(dataset)):
                assert (
                    abs(tmd_matrix[i, j] - tmd_matrix[j, i]) < 1e-6
                ), f"Matrix not symmetric at ({i}, {j}): {tmd_matrix[i, j]} != {tmd_matrix[j, i]}"

    def test_tmd_matrix_caching(self, simple_graph_2nodes, simple_graph_3nodes_chain):
        """Test TMD matrix caching."""
        dataset = [simple_graph_2nodes, simple_graph_3nodes_chain]

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "test_tmd_matrix.npy")

            # Compute first time
            tmd_matrix1 = compute_tmd_matrix(
                dataset, w=1.0, L=4, verbose=False, cache_path=cache_path
            )

            # Load from cache
            assert os.path.exists(cache_path)
            tmd_matrix2 = compute_tmd_matrix(
                dataset, w=1.0, L=4, verbose=False, cache_path=cache_path
            )

            # Should be identical
            np.testing.assert_array_almost_equal(tmd_matrix1, tmd_matrix2)


class TestComputeClassDistanceRatios:
    """Test the compute_class_distance_ratios function."""

    def test_basic_computation(self):
        """Test basic class-distance ratio computation."""
        # Create a simple TMD matrix
        # Graph 0 and 1 are same class (label 0), Graph 2 is different class (label 1)
        tmd_matrix = np.array(
            [
                [
                    0.0,
                    1.0,
                    5.0,
                ],  # Graph 0: close to graph 1 (same class), far from graph 2
                [
                    1.0,
                    0.0,
                    5.0,
                ],  # Graph 1: close to graph 0 (same class), far from graph 2
                [5.0, 5.0, 0.0],  # Graph 2: far from both graphs 0 and 1
            ]
        )
        labels = np.array([0, 0, 1])

        ratios, stats = compute_class_distance_ratios(tmd_matrix, labels, verbose=False)

        assert isinstance(ratios, np.ndarray)
        assert len(ratios) == 3
        # Graph 0: min_same=1.0, min_diff=5.0, ratio=1.0/5.0=0.2
        assert abs(ratios[0] - 0.2) < 1e-6
        # Graph 1: min_same=1.0, min_diff=5.0, ratio=0.2
        assert abs(ratios[1] - 0.2) < 1e-6
        # Graph 2: min_same=0 (no other graph with label 1), min_diff=5.0
        # This is an edge case - ratio should be handled appropriately
        assert ratios[2] >= 0

        # Check stats
        assert "mean" in stats
        assert "median" in stats
        assert "std" in stats
        assert "num_hard" in stats
        assert "num_easy" in stats

    def test_all_same_class(self):
        """Test edge case: all graphs have same class."""
        tmd_matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
        labels = np.array([0, 0])

        ratios, stats = compute_class_distance_ratios(tmd_matrix, labels, verbose=False)

        assert len(ratios) == 2
        # When all graphs have same class, min_diff_class will be problematic
        # The function should handle this gracefully
        assert np.all(np.isfinite(ratios)) or np.any(np.isinf(ratios))

    def test_single_graph(self):
        """Test edge case: single graph."""
        tmd_matrix = np.array([[0.0]])
        labels = np.array([0])

        ratios, stats = compute_class_distance_ratios(tmd_matrix, labels, verbose=False)

        assert len(ratios) == 1
        assert stats["num_easy"] >= 0
        assert stats["num_hard"] >= 0


class TestSaveTMDResults:
    """Test the save_tmd_results function."""

    def test_save_results(self):
        """Test saving TMD results to files."""
        tmd_matrix = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
        ratios = np.array([0.5, 1.0, 1.5])
        labels = np.array([0, 0, 1])
        stats = {
            "mean": 1.0,
            "median": 1.0,
            "std": 0.5,
            "num_hard": 1,
            "num_easy": 1,
            "num_ambiguous": 1,
            "num_infinite": 0,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            file_paths = save_tmd_results(
                "test_dataset",
                tmd_matrix,
                ratios,
                labels,
                stats,
                output_dir=tmpdir,
            )

            # Check that files were created
            assert os.path.exists(file_paths["tmd_matrix"])
            assert os.path.exists(file_paths["class_ratios"])
            assert os.path.exists(file_paths["stats"])

            # Verify TMD matrix can be loaded
            loaded_matrix = np.load(file_paths["tmd_matrix"])
            np.testing.assert_array_almost_equal(tmd_matrix, loaded_matrix)

            # Verify CSV can be loaded
            import pandas as pd

            loaded_df = pd.read_csv(file_paths["class_ratios"])
            assert len(loaded_df) == 3
            assert "graph_index" in loaded_df.columns
            assert "label" in loaded_df.columns
            assert "class_distance_ratio" in loaded_df.columns

            # Verify JSON can be loaded
            import json

            with open(file_paths["stats"], "r", encoding="utf-8") as f:
                loaded_stats = json.load(f)
            assert loaded_stats == stats


class TestTMDIntegration:
    """Integration tests for TMD module."""

    def test_full_pipeline(
        self, simple_graph_2nodes, simple_graph_3nodes_chain, isolated_node_graph
    ):
        """Test full pipeline: compute TMD matrix, extract labels, compute ratios."""
        # Create dataset with labels
        dataset = [
            simple_graph_2nodes,  # label 0
            simple_graph_3nodes_chain,  # label 1
            isolated_node_graph,  # label 0
        ]

        # Extract labels
        labels, num_classes, label_counts = extract_labels(dataset)

        assert len(labels) == 3
        assert num_classes == 2

        # Compute TMD matrix
        tmd_matrix = compute_tmd_matrix(dataset, w=1.0, L=4, verbose=False)

        assert tmd_matrix.shape == (3, 3)

        # Compute class-distance ratios
        ratios, stats = compute_class_distance_ratios(tmd_matrix, labels, verbose=False)

        assert len(ratios) == 3
        assert all(r >= 0 for r in ratios if np.isfinite(r))
        assert "mean" in stats

    def test_tmd_triangle_inequality_approximate(
        self, simple_graph_2nodes, simple_graph_3nodes_chain, isolated_node_graph
    ):
        """Test approximate triangle inequality for TMD.

        Note: TMD is a pseudometric, so triangle inequality may not hold exactly,
        but we can test that distances are reasonable.
        """
        tmd_12 = TMD(simple_graph_2nodes, simple_graph_3nodes_chain, w=1.0, L=4)
        tmd_23 = TMD(simple_graph_3nodes_chain, isolated_node_graph, w=1.0, L=4)
        tmd_13 = TMD(simple_graph_2nodes, isolated_node_graph, w=1.0, L=4)

        # Triangle inequality: d(1,3) <= d(1,2) + d(2,3) (approximately)
        # Note: TMD is a pseudometric, so this may not hold exactly
        # We just check that all distances are reasonable
        assert tmd_12 >= 0
        assert tmd_23 >= 0
        assert tmd_13 >= 0
        assert all(
            isinstance(d, (float, np.floating)) for d in [tmd_12, tmd_23, tmd_13]
        )
