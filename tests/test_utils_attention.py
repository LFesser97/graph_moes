"""Test suite for orthogonal matrix generation in attention module.

This module tests the orthogonal_matrix function that generates orthogonal matrices
used in the Performer attention mechanism. Orthogonal matrices have the property that
Q @ Q.T = I (identity matrix), which is useful for maintaining orthogonality constraints
in neural network layers.

These tests serve as documentation for how orthogonal matrices are generated and
validated.
"""

import torch

from graph_moes.utils.attention import orthogonal_matrix


class TestOrthogonalMatrix:
    """Test the orthogonal_matrix function."""

    def test_square_matrix_orthogonality(self):
        """Test that square orthogonal matrices satisfy Q @ Q.T = I.

        For a square orthogonal matrix Q, we should have Q @ Q.T = I (identity matrix).
        This is the fundamental property of orthogonal matrices.
        """
        num_rows = 10
        num_cols = 10  # Square matrix
        Q = orthogonal_matrix(num_rows, num_cols)

        # Check shape
        assert Q.shape == (num_rows, num_cols)

        # Check orthogonality: Q @ Q.T should be close to identity
        identity = torch.eye(num_cols)
        product = Q @ Q.T

        # Should be close to identity matrix (allow small numerical errors)
        assert torch.allclose(product, identity, atol=1e-5)

    def test_square_matrix_shape(self):
        """Test that square matrix has correct shape."""
        num_rows = 5
        num_cols = 5
        Q = orthogonal_matrix(num_rows, num_cols)

        assert Q.shape == (num_rows, num_cols)
        assert Q.shape[0] == Q.shape[1]  # Square

    def test_rectangular_matrix_shape(self):
        """Test that rectangular matrices have correct shape.

        Orthogonal matrices can be rectangular (tall or wide).
        """
        # Tall matrix (more rows than columns)
        num_rows = 10
        num_cols = 5
        Q = orthogonal_matrix(num_rows, num_cols)

        assert Q.shape == (num_rows, num_cols)
        assert Q.shape[0] > Q.shape[1]

        # Wide matrix (more columns than rows)
        num_rows = 5
        num_cols = 10
        Q = orthogonal_matrix(num_rows, num_cols)

        assert Q.shape == (num_rows, num_cols)
        assert Q.shape[0] < Q.shape[1]

    def test_rectangular_matrix_properties(self):
        """Test properties of rectangular orthogonal matrices.

        For rectangular matrices Q with shape (m, n):
        - The implementation uses blocks, so tall matrices have scaled Q.T @ Q
        - Wide matrices: Q @ Q.T should be identity (rows are orthonormal)
        """
        # Wide matrix: rows should be orthonormal
        num_rows = 5
        num_cols = 10
        Q = orthogonal_matrix(num_rows, num_cols)

        # Q @ Q.T should be identity (rows are orthonormal)
        identity = torch.eye(num_rows)
        product = Q @ Q.T
        assert torch.allclose(product, identity, atol=1e-5)

        # Tall matrix: implementation uses blocks, so Q.T @ Q is scaled
        # (not exactly identity, but structured)
        num_rows = 10
        num_cols = 5
        Q = orthogonal_matrix(num_rows, num_cols)

        # Q.T @ Q should be approximately a scaled identity (due to block structure)
        product = Q.T @ Q
        # Diagonal should be approximately constant (block structure)
        diag_values = torch.diag(product)
        # Should be approximately constant (within reasonable tolerance)
        assert torch.std(diag_values) < 0.1  # Low variance

    def test_small_matrix(self):
        """Test with very small matrix dimensions."""
        Q = orthogonal_matrix(2, 2)

        assert Q.shape == (2, 2)
        identity = torch.eye(2)
        assert torch.allclose(Q @ Q.T, identity, atol=1e-5)

    def test_large_matrix(self):
        """Test with larger matrix dimensions."""
        num_rows = 50
        num_cols = 50
        Q = orthogonal_matrix(num_rows, num_cols)

        assert Q.shape == (num_rows, num_cols)
        identity = torch.eye(num_cols)
        # Allow slightly larger tolerance for larger matrices
        assert torch.allclose(Q @ Q.T, identity, atol=1e-4)

    def test_different_ratios(self):
        """Test matrices with different row/column ratios."""
        # 1:2 ratio (wide matrix) - rows are orthonormal
        Q = orthogonal_matrix(10, 20)
        assert Q.shape == (10, 20)
        assert torch.allclose(Q @ Q.T, torch.eye(10), atol=1e-5)

        # 2:1 ratio (tall matrix) - uses blocks, Q.T @ Q is scaled
        Q = orthogonal_matrix(20, 10)
        assert Q.shape == (20, 10)
        # Q.T @ Q should have diagonal approximately constant (block structure)
        product = Q.T @ Q
        diag_values = torch.diag(product)
        assert torch.std(diag_values) < 0.1  # Low variance

        # 3:1 ratio (tall matrix)
        Q = orthogonal_matrix(30, 10)
        assert Q.shape == (30, 10)
        product = Q.T @ Q
        diag_values = torch.diag(product)
        assert torch.std(diag_values) < 0.1  # Low variance

    def test_tall_matrix_block_structure(self):
        """Test that tall matrices use block structure.

        For tall matrix Q (m > n), the implementation uses blocks.
        The columns are structured but may not be exactly orthonormal
        due to the block concatenation approach.
        """
        num_rows = 15
        num_cols = 5
        Q = orthogonal_matrix(num_rows, num_cols)

        # Q.T @ Q should have a structured form (approximately scaled identity)
        product = Q.T @ Q
        # Diagonal should be approximately constant
        diag_values = torch.diag(product)
        assert torch.std(diag_values) < 0.1  # Low variance indicates structure

        # Off-diagonal should be small (columns are approximately orthogonal)
        # Extract off-diagonal elements
        mask = ~torch.eye(num_cols, dtype=torch.bool)
        off_diag = product[mask]
        # Off-diagonal should be close to zero
        assert torch.max(torch.abs(off_diag)) < 0.5  # Reasonable tolerance

    def test_rows_orthonormal_wide_matrix(self):
        """Test that rows of wide matrix are orthonormal.

        For wide matrix Q (m < n), the rows should be orthonormal:
        - Each row has unit norm
        - Rows are orthogonal to each other
        """
        num_rows = 5
        num_cols = 15
        Q = orthogonal_matrix(num_rows, num_cols)

        # Check that each row has unit norm
        for i in range(num_rows):
            row_norm = torch.norm(Q[i, :])
            assert torch.allclose(row_norm, torch.tensor(1.0), atol=1e-5)

        # Check that rows are orthogonal
        for i in range(num_rows):
            for j in range(i + 1, num_rows):
                dot_product = torch.dot(Q[i, :], Q[j, :])
                assert torch.allclose(dot_product, torch.tensor(0.0), atol=1e-5)

    def test_determinant_square_matrix(self):
        """Test that square orthogonal matrices have determinant ±1.

        For square orthogonal matrices, det(Q) = ±1 (not necessarily +1).
        This is because orthogonal matrices preserve lengths.
        """
        Q = orthogonal_matrix(5, 5)
        det = torch.det(Q)

        # Determinant should be close to 1 or -1
        assert torch.allclose(torch.abs(det), torch.tensor(1.0), atol=1e-5)

    def test_different_calls_produce_different_matrices(self):
        """Test that multiple calls produce different (random) matrices.

        Each call should generate a new random orthogonal matrix, so consecutive
        calls should produce different matrices (with high probability).
        """
        Q1 = orthogonal_matrix(10, 10)
        Q2 = orthogonal_matrix(10, 10)

        # Matrices should be different (not identical)
        assert not torch.allclose(Q1, Q2, atol=1e-10)

        # But both should still be orthogonal
        identity = torch.eye(10)
        assert torch.allclose(Q1 @ Q1.T, identity, atol=1e-5)
        assert torch.allclose(Q2 @ Q2.T, identity, atol=1e-5)

    def test_tensor_type(self):
        """Test that returned tensor has correct type."""
        Q = orthogonal_matrix(5, 5)

        assert isinstance(Q, torch.Tensor)
        assert Q.dtype in (torch.float32, torch.float64)

    def test_device_compatibility(self):
        """Test that matrix is on CPU (as implementation uses .cpu())."""
        Q = orthogonal_matrix(5, 5)

        # Implementation uses .cpu() explicitly, so should be on CPU
        assert Q.device.type == "cpu"

    def test_edge_case_one_by_one(self):
        """Test edge case: 1x1 matrix."""
        Q = orthogonal_matrix(1, 1)

        assert Q.shape == (1, 1)
        # 1x1 orthogonal matrix is just ±1
        assert torch.allclose(torch.abs(Q), torch.tensor(1.0), atol=1e-5)

    def test_edge_case_one_column(self):
        """Test edge case: n×1 matrix (single column).

        For tall matrices with one column, the block structure means
        the column norm is approximately sqrt(num_blocks), not 1.
        """
        Q = orthogonal_matrix(10, 1)

        assert Q.shape == (10, 1)
        # With block structure (10 blocks of 1x1), column norm is approximately sqrt(10)
        # But each block has unit norm, so concatenated vector has norm sqrt(10)
        col_norm = torch.norm(Q)
        # Should be approximately sqrt(10) ≈ 3.16
        assert torch.allclose(col_norm, torch.tensor(3.16), atol=0.1)

    def test_edge_case_one_row(self):
        """Test edge case: 1×n matrix (single row)."""
        Q = orthogonal_matrix(1, 10)

        assert Q.shape == (1, 10)
        # Single row should have unit norm
        assert torch.allclose(torch.norm(Q), torch.tensor(1.0), atol=1e-5)
