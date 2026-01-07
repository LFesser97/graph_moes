"""Test suite for helper functions in track_avg_accuracy module.

This module tests utility functions used for tracking and plotting average accuracy,
specifically the get_detailed_model_name function that generates model names for
both standard GNN models and Mixture of Experts (MoE) models.

These tests serve as documentation for how model naming conventions work.
"""

from graph_moes.experiments.track_avg_accuracy import get_detailed_model_name


class TestGetDetailedModelName:
    """Test the get_detailed_model_name function."""

    def test_non_moe_model_gcn(self):
        """Test that non-MoE GCN model returns just the layer type."""
        result = get_detailed_model_name("GCN", layer_types=None, router_type="MLP")
        assert result == "GCN"
        # Router type should be ignored for non-MoE models
        result2 = get_detailed_model_name("GCN", layer_types=None, router_type="GNN")
        assert result2 == "GCN"

    def test_non_moe_model_gin(self):
        """Test that non-MoE GIN model returns just the layer type."""
        result = get_detailed_model_name("GIN", layer_types=None, router_type="MLP")
        assert result == "GIN"

    def test_non_moe_model_sage(self):
        """Test that non-MoE SAGE model returns just the layer type."""
        result = get_detailed_model_name("SAGE", layer_types=None, router_type="MLP")
        assert result == "SAGE"

    def test_moe_model_with_mlp_router(self):
        """Test that MoE model with MLP router includes router type and experts."""
        result = get_detailed_model_name(
            "MoE", layer_types=["GCN", "GIN"], router_type="MLP"
        )
        assert result == "MoE_MLP_GCN_GIN"
        # Order should be preserved
        assert "GCN_GIN" in result
        assert "MLP" in result

    def test_moe_model_with_gnn_router(self):
        """Test that MoE model with GNN router includes router type and experts."""
        result = get_detailed_model_name(
            "MoE", layer_types=["GCN", "GIN"], router_type="GNN"
        )
        assert result == "MoE_GNN_GCN_GIN"
        assert "GNN" in result
        assert "GCN_GIN" in result

    def test_moe_e_model(self):
        """Test that MoE_E model includes router type and experts."""
        result = get_detailed_model_name(
            "MoE_E", layer_types=["GIN", "Unitary"], router_type="MLP"
        )
        assert result == "MoE_E_MLP_GIN_Unitary"
        assert "MoE_E" in result
        assert "GIN_Unitary" in result

    def test_moe_model_expert_order_preserved(self):
        """Test that expert order in layer_types is preserved in the name."""
        result1 = get_detailed_model_name(
            "MoE", layer_types=["GCN", "GIN"], router_type="MLP"
        )
        result2 = get_detailed_model_name(
            "MoE", layer_types=["GIN", "GCN"], router_type="MLP"
        )
        # Should be different due to order
        assert result1 == "MoE_MLP_GCN_GIN"
        assert result2 == "MoE_MLP_GIN_GCN"
        assert result1 != result2

    def test_moe_model_multiple_experts(self):
        """Test that MoE model with multiple expert types works correctly."""
        result = get_detailed_model_name(
            "MoE", layer_types=["GCN", "SAGE", "GAT"], router_type="MLP"
        )
        # Should join all expert types with underscores
        assert result == "MoE_MLP_GCN_SAGE_GAT"
        assert "GCN" in result
        assert "SAGE" in result
        assert "GAT" in result

    def test_moe_model_default_router_type(self):
        """Test that default router_type 'MLP' works correctly."""
        result = get_detailed_model_name(
            "MoE", layer_types=["GCN", "GIN"], router_type="MLP"
        )
        assert result == "MoE_MLP_GCN_GIN"

    def test_format_structure(self):
        """Test that the format is consistent: layer_type_router_expert1_expert2."""
        # Non-MoE: just layer_type
        result = get_detailed_model_name("GCN", layer_types=None, router_type="MLP")
        assert (
            "_" not in result or result.count("_") == 0
        )  # No underscores for simple names

        # MoE: layer_type_router_expert1_expert2
        result = get_detailed_model_name(
            "MoE", layer_types=["GCN", "GIN"], router_type="MLP"
        )
        parts = result.split("_")
        assert len(parts) == 4  # MoE, MLP, GCN, GIN
        assert parts[0] == "MoE"
        assert parts[1] == "MLP"
        assert parts[2] == "GCN"
        assert parts[3] == "GIN"

    def test_edge_case_empty_layer_types_list(self):
        """Test behavior with empty layer_types list (edge case)."""
        # Empty list should still create MoE format, but with no experts
        result = get_detailed_model_name("MoE", layer_types=[], router_type="MLP")
        assert result == "MoE_MLP_"  # Empty string after final underscore
        # Note: This might not be a realistic use case, but tests the function's behavior

    def test_edge_case_single_expert(self):
        """Test MoE with single expert (edge case - MoE typically uses 2 experts)."""
        result = get_detailed_model_name("MoE", layer_types=["GCN"], router_type="MLP")
        assert result == "MoE_MLP_GCN"
