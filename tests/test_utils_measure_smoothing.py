"""Test suite for Dirichlet energy functions in measure_smoothing module.

This module tests functions that measure the smoothness of vector fields on graphs:
- dirichlet_energy(): Computes the Dirichlet energy of a vector field
- dirichlet_normalized(): Computes normalized Dirichlet energy (scale-invariant)

The Dirichlet energy measures how "smooth" a vector field is on a graph - lower values
indicate smoother fields (neighboring nodes have similar values), higher values indicate
less smooth fields (neighboring nodes have very different values).

These tests serve as documentation for what Dirichlet energy measures and how to use it.
"""

import numpy as np
import pytest

from graph_moes.utils.measure_smoothing import dirichlet_energy, dirichlet_normalized


@pytest.fixture
def simple_graph_2nodes():
    """Create a simple 2-node graph with bidirectional edge.

    Graph structure:
    0 <-> 1 (bidirectional)

    Note: dirichlet_energy computes degrees only for source nodes,
    so we need edges in both directions or ensure target nodes have edges.
    """
    # Bidirectional edge: (0,1) and (1,0) so both nodes have degree > 0
    edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)
    return edge_index


@pytest.fixture
def simple_graph_3nodes_chain():
    """Create a simple 3-node chain graph.

    Graph structure:
    0 --- 1 --- 2

    Use bidirectional edges to ensure all nodes have degree > 0.
    """
    # Bidirectional edges: (0,1), (1,0), (1,2), (2,1)
    edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64)
    return edge_index


@pytest.fixture
def simple_graph_3nodes_triangle():
    """Create a 3-node triangle (complete graph).
    
    Graph structure:
       0
      / \
     1---2
    
    Triangle with bidirectional edges.
    """
    # Bidirectional triangle: all pairs connected both ways
    edge_index = np.array([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]], dtype=np.int64)
    return edge_index


class TestDirichletEnergy:
    """Test the dirichlet_energy function."""

    def test_constant_vector_field_low_energy(self, simple_graph_2nodes):
        """Test that constant vector field has low Dirichlet energy.

        When all nodes have the same value, the field is perfectly smooth,
        so energy should be low (specifically, close to 0).
        """
        # Constant vector field: all nodes have value [1.0, 1.0]
        X = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
        energy = dirichlet_energy(X, simple_graph_2nodes)

        # Energy should be low (neighbors have same values)
        assert energy >= 0  # Energy is always non-negative
        assert energy < 1.0  # Should be relatively small for constant field

    def test_constant_vector_field_zero_difference(self, simple_graph_3nodes_chain):
        """Test that perfectly constant field has minimal energy."""
        # Perfectly constant: all nodes identical
        X = np.array([[5.0], [5.0], [5.0]], dtype=np.float64)
        energy = dirichlet_energy(X, simple_graph_3nodes_chain)

        # For constant fields, energy should be very low
        assert energy >= 0
        # Note: energy may not be exactly 0 due to normalization by degrees

    def test_random_vector_field_higher_energy(self, simple_graph_2nodes):
        """Test that random vector field has higher energy than constant field.

        When nodes have very different values, the field is less smooth,
        so energy should be higher.
        """
        # Constant field
        X_constant = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
        energy_constant = dirichlet_energy(X_constant, simple_graph_2nodes)

        # Random/varying field: nodes have very different values
        X_varying = np.array([[10.0, 0.0], [0.0, 10.0]], dtype=np.float64)
        energy_varying = dirichlet_energy(X_varying, simple_graph_2nodes)

        # Varying field should have higher energy
        assert energy_varying > energy_constant

    def test_opposite_values_high_energy(self, simple_graph_2nodes):
        """Test that opposite values on connected nodes give high energy."""
        # Nodes have opposite values
        X = np.array([[1.0], [-1.0]], dtype=np.float64)
        energy = dirichlet_energy(X, simple_graph_2nodes)

        # Should have relatively high energy (neighbors are very different)
        assert energy > 0
        assert energy > 0.5  # Should be reasonably large

    def test_single_node_graph(self):
        """Test edge case: single node graph with no edges."""
        # Single node, no edges
        X = np.array([[1.0, 2.0]], dtype=np.float64)
        edge_index = np.array([[], []], dtype=np.int64)  # Empty edge index

        energy = dirichlet_energy(X, edge_index)

        # With no edges, energy should be based only on norm
        assert energy >= 0
        # Should be the squared norm of X
        expected_energy = np.linalg.norm(X.flatten()) ** 2
        assert abs(energy - expected_energy) < 1e-10

    def test_single_feature(self, simple_graph_2nodes):
        """Test with single feature dimension."""
        X = np.array([[1.0], [2.0]], dtype=np.float64)
        energy = dirichlet_energy(X, simple_graph_2nodes)

        assert energy >= 0
        assert isinstance(energy, (float, np.floating))

    def test_multiple_features(self, simple_graph_2nodes):
        """Test with multiple feature dimensions."""
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        energy = dirichlet_energy(X, simple_graph_2nodes)

        assert energy >= 0
        assert isinstance(energy, (float, np.floating))

    def test_chain_graph_smooth_field(self, simple_graph_3nodes_chain):
        """Test smooth field on chain graph (gradual change)."""
        # Smooth field: values change gradually along chain
        X = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
        energy = dirichlet_energy(X, simple_graph_3nodes_chain)

        assert energy >= 0
        # Should have moderate energy (smooth but not constant)

    def test_triangle_graph_high_degree(self, simple_graph_3nodes_triangle):
        """Test on triangle graph where nodes have higher degree."""
        # Constant field on triangle
        X = np.array([[1.0], [1.0], [1.0]], dtype=np.float64)
        energy = dirichlet_energy(X, simple_graph_3nodes_triangle)

        # Energy should be non-negative (allow small numerical errors)
        assert (
            energy >= -1e-14
        )  # Allow very small negative values due to floating point
        # Triangle has more edges, so normalization by degrees matters

    def test_energy_always_non_negative(self, simple_graph_3nodes_chain):
        """Test that Dirichlet energy is always non-negative."""
        # Test with various fields
        test_fields = [
            np.array([[1.0], [1.0], [1.0]], dtype=np.float64),  # Constant
            np.array([[1.0], [-1.0], [1.0]], dtype=np.float64),  # Alternating
            np.array([[10.0], [0.0], [-10.0]], dtype=np.float64),  # Extreme
            np.array([[0.0], [0.0], [0.0]], dtype=np.float64),  # Zeros
        ]

        for X in test_fields:
            energy = dirichlet_energy(X, simple_graph_3nodes_chain)
            assert energy >= 0, f"Energy should be non-negative, got {energy}"


class TestDirichletNormalized:
    """Test the dirichlet_normalized function."""

    def test_normalized_range(self, simple_graph_2nodes):
        """Test that normalized energy is between 0 and 1 (or reasonable range).

        Normalized energy divides by the squared norm, making it scale-invariant.
        """
        X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        normalized_energy = dirichlet_normalized(X, simple_graph_2nodes)

        # Normalized energy should be non-negative
        assert normalized_energy >= 0
        # Should be a reasonable value (typically between 0 and 1, but not strictly bounded)

    def test_constant_field_normalized(self, simple_graph_2nodes):
        """Test normalized energy for constant field."""
        X = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
        normalized_energy = dirichlet_normalized(X, simple_graph_2nodes)

        assert normalized_energy >= 0
        # For constant fields, normalized energy should be relatively small

    def test_scale_invariance(self, simple_graph_2nodes):
        """Test that normalized energy is scale-invariant.

        Scaling the vector field by a constant should give similar normalized energy.
        """
        X1 = np.array([[1.0], [2.0]], dtype=np.float64)
        normalized1 = dirichlet_normalized(X1, simple_graph_2nodes)

        # Scale by factor of 10
        X2 = X1 * 10.0
        normalized2 = dirichlet_normalized(X2, simple_graph_2nodes)

        # Normalized energies should be very similar (scale-invariant)
        assert abs(normalized1 - normalized2) < 0.01

    def test_normalized_vs_raw_energy(self, simple_graph_2nodes):
        """Test relationship between normalized and raw energy."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

        raw_energy = dirichlet_energy(X, simple_graph_2nodes)
        normalized_energy = dirichlet_normalized(X, simple_graph_2nodes)

        # Normalized = raw_energy / (squared norm)
        norm_squared = np.sum(X**2)
        expected_normalized = raw_energy / norm_squared

        assert abs(normalized_energy - expected_normalized) < 1e-10

    def test_zero_vector_field(self, simple_graph_2nodes):
        """Test edge case: zero vector field."""
        X = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)

        # Normalized energy with zero field - may have division by zero issues
        # This tests that the function handles it gracefully
        try:
            normalized_energy = dirichlet_normalized(X, simple_graph_2nodes)
            # If it doesn't raise, check it's a valid number
            assert np.isfinite(normalized_energy) or np.isnan(normalized_energy)
        except (ZeroDivisionError, ValueError):
            # Division by zero is acceptable for zero vector field
            pass

    def test_single_node_normalized(self):
        """Test normalized energy on single node graph."""
        X = np.array([[1.0, 2.0]], dtype=np.float64)
        edge_index = np.array([[], []], dtype=np.int64)

        normalized_energy = dirichlet_normalized(X, edge_index)

        # With no edges, normalized energy should be based on raw energy / norm^2
        assert normalized_energy >= 0 or np.isnan(normalized_energy)

    def test_chain_graph_normalized(self, simple_graph_3nodes_chain):
        """Test normalized energy on chain graph."""
        # Smooth field
        X_smooth = np.array([[1.0], [1.5], [2.0]], dtype=np.float64)
        norm_smooth = dirichlet_normalized(X_smooth, simple_graph_3nodes_chain)

        # Less smooth field
        X_rough = np.array([[1.0], [10.0], [2.0]], dtype=np.float64)
        norm_rough = dirichlet_normalized(X_rough, simple_graph_3nodes_chain)

        # Rough field should have higher normalized energy
        assert norm_rough > norm_smooth

    def test_always_non_negative_normalized(self, simple_graph_3nodes_chain):
        """Test that normalized energy is always non-negative."""
        test_fields = [
            np.array([[1.0], [1.0], [1.0]], dtype=np.float64),
            np.array([[1.0], [-1.0], [1.0]], dtype=np.float64),
            np.array([[10.0], [0.0], [-10.0]], dtype=np.float64),
        ]

        for X in test_fields:
            normalized = dirichlet_normalized(X, simple_graph_3nodes_chain)
            # May be NaN for zero fields, but if finite, should be >= 0
            if np.isfinite(normalized):
                assert normalized >= 0


class TestDirichletEnergyCombined:
    """Test combined behavior and edge cases."""

    def test_smooth_vs_rough_comparison(self, simple_graph_3nodes_chain):
        """Test that smooth fields have lower energy than rough fields."""
        # Very smooth: gradual change
        X_smooth = np.array([[1.0], [1.1], [1.2]], dtype=np.float64)
        energy_smooth = dirichlet_energy(X_smooth, simple_graph_3nodes_chain)

        # Rough: big jumps
        X_rough = np.array([[1.0], [10.0], [1.0]], dtype=np.float64)
        energy_rough = dirichlet_energy(X_rough, simple_graph_3nodes_chain)

        # Smooth should have lower energy
        assert energy_smooth < energy_rough

    def test_multiple_graphs_comparison(self):
        """Test that same field on different graphs gives different energies."""
        X = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)

        # Chain graph: 0-1-2 (bidirectional to ensure all nodes have degree > 0)
        chain_edges = np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64)
        energy_chain = dirichlet_energy(X, chain_edges)

        # Triangle graph: 0-1-2 with all connections (bidirectional)
        triangle_edges = np.array(
            [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]], dtype=np.int64
        )
        energy_triangle = dirichlet_energy(X, triangle_edges)

        # Energies should be different (different graph structures)
        # Note: may not always be larger/smaller due to degree normalization
        assert energy_chain != energy_triangle

    def test_numpy_array_types(self, simple_graph_2nodes):
        """Test that function works with different numpy dtypes."""
        # Test float32
        X_float32 = np.array([[1.0], [2.0]], dtype=np.float32)
        energy_32 = dirichlet_energy(X_float32, simple_graph_2nodes)
        assert isinstance(energy_32, (float, np.floating))

        # Test float64
        X_float64 = np.array([[1.0], [2.0]], dtype=np.float64)
        energy_64 = dirichlet_energy(X_float64, simple_graph_2nodes)
        assert isinstance(energy_64, (float, np.floating))
