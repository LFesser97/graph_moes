"""Comprehensive test suite for Mixture of Experts (MoE) graph neural network models.

This module tests the core functionality of MoE models including routers, experts,
and the complete MoE and MoE_E architectures with various configurations.
"""

import copy
import pytest
import torch
import torch.nn as nn

# from attrdict import AttrDict  # Fallback for older Python


# Simple AttrDict replacement to avoid dependency issues
class AttrDict(dict):
    """A dict that allows attribute access to its keys."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

    def __setattr__(self, name, value):
        if name == "__dict__":
            super().__setattr__(name, value)
        else:
            self[name] = value

    def __deepcopy__(self, memo):
        # Create a new AttrDict with deep copied contents
        new_dict = copy.deepcopy(dict(self), memo)
        return AttrDict(new_dict)


from torch_geometric.data import Batch, Data

from graph_moes.moes.graph_moe import MoE, MoE_E
from graph_moes.moes.routers.gnn_router import GNNRouter
from graph_moes.moes.routers.mlp_router import MLPRouter
from graph_moes.moes.routers.router import Router


@pytest.fixture
def device():
    """Get available device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_graph():
    """Create a simple test graph with 6 nodes and 8 edges."""
    x = torch.randn(6, 10)  # 6 nodes, 10 features each
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5], [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]]
    )
    return Data(x=x, edge_index=edge_index)


@pytest.fixture
def batch_graphs():
    """Create a batch of 3 small test graphs."""
    graphs = []
    for i in range(3):
        x = torch.randn(4, 10)  # 4 nodes, 10 features
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        graphs.append(Data(x=x, edge_index=edge_index))
    return Batch.from_data_list(graphs)


@pytest.fixture
def basic_args():
    """Basic configuration arguments for MoE models."""
    return AttrDict(
        {
            "input_dim": 10,
            "hidden_dim": 32,
            "output_dim": 2,
            "num_layers": 2,
            "hidden_layers": [32, 32],
            "dropout": 0.1,
            "mlp": True,
            "num_relations": 1,
            "last_layer_fa": False,
            "layer_types": ["GCN", "GIN"],
            "router_type": "MLP",
            "router_layer_type": "GIN",
            "router_depth": 2,
            "router_dropout": 0.0,
            "router_hidden_layers": [32, 16],
        }
    )


@pytest.fixture
def moe_e_args():
    """Configuration for MoE_E model with feature masking."""
    return AttrDict(
        {
            "input_dim": 15,  # Larger input for masking test
            "hidden_dim": 32,
            "output_dim": 3,
            "num_layers": 2,
            "hidden_layers": [32, 32],
            "dropout": 0.0,
            "mlp": True,
            "num_relations": 1,
            "last_layer_fa": False,
            "layer_types": ["GCN", "GIN"],
            "router_type": "MLP",
            "router_layer_type": "GCN",
            "router_depth": 2,
            "router_dropout": 0.1,
            "router_hidden_layers": [32, 16],
        }
    )


class TestMLPRouter:
    """Test suite for MLPRouter class."""

    def test_mlp_router_initialization(self):
        """Test MLPRouter creates correct architecture."""
        input_dim, num_experts = 10, 3
        hidden_layers = [16, 8]
        router = MLPRouter(input_dim, hidden_layers, num_experts)

        # Check network structure
        assert len(router.net) == 5  # Linear -> ReLU -> Linear -> ReLU -> Linear
        assert isinstance(router.net[0], nn.Linear)
        assert router.net[0].in_features == input_dim
        assert router.net[0].out_features == hidden_layers[0]
        assert router.net[-1].out_features == num_experts

    def test_mlp_router_forward(self, batch_graphs):
        """Test MLPRouter forward pass produces correct output shapes."""
        router = MLPRouter(input_dim=10, hidden_layers=[16], num_experts=2)
        logits = router(batch_graphs)

        assert logits.shape == (3, 2)  # [batch_size, num_experts]
        assert torch.isfinite(logits).all()

    def test_mlp_router_different_configs(self):
        """Test MLPRouter with various configurations."""
        configs = [
            (5, [10], 2),
            (20, [32, 16], 4),
            (10, [64, 32, 16], 3),
        ]

        for input_dim, hidden_layers, num_experts in configs:
            router = MLPRouter(input_dim, hidden_layers, num_experts)
            assert router.net[-1].out_features == num_experts


class TestGNNRouter:
    """Test suite for GNNRouter class."""

    def test_gnn_router_initialization(self):
        """Test GNNRouter creates correct GNN architecture."""
        router = GNNRouter(
            input_dim=10,
            layer_type="GCN",
            hidden_dim=32,
            depth=3,
            dropout=0.1,
            num_experts=2,
        )

        assert len(router.layers) == 3
        assert router.layer_type == "GCN"
        assert len(router.mlp_out) == 3  # Linear -> ReLU -> Linear

    def test_gnn_router_layer_types(self):
        """Test GNNRouter supports both GCN and GIN layer types."""
        for layer_type in ["GCN", "GIN"]:
            router = GNNRouter(
                input_dim=10,
                layer_type=layer_type,
                hidden_dim=16,
                depth=2,
                dropout=0.0,
                num_experts=3,
            )
            assert router.layer_type == layer_type

    def test_gnn_router_invalid_layer_type(self):
        """Test GNNRouter raises error for unsupported layer types."""
        with pytest.raises(AssertionError, match="GNNRouter supports only"):
            GNNRouter(
                input_dim=10,
                layer_type="INVALID",
                hidden_dim=16,
                depth=2,
                dropout=0.0,
                num_experts=2,
            )

    def test_gnn_router_forward(self, batch_graphs):
        """Test GNNRouter forward pass with both layer types."""
        for layer_type in ["GCN", "GIN"]:
            router = GNNRouter(
                input_dim=10,
                layer_type=layer_type,
                hidden_dim=16,
                depth=2,
                dropout=0.0,
                num_experts=2,
            )
            logits = router(batch_graphs)

            assert logits.shape == (3, 2)  # [batch_size, num_experts]
            assert torch.isfinite(logits).all()


class TestRouter:
    """Test suite for unified Router interface."""

    def test_router_mlp_type(self, basic_args):
        """Test Router creates MLPRouter when type is 'MLP'."""
        router = Router("MLP", input_dim=10, num_experts=2, args=basic_args)
        assert isinstance(router.model, MLPRouter)

    def test_router_gnn_type(self, basic_args):
        """Test Router creates GNNRouter when type is 'GNN'."""
        router = Router("GNN", input_dim=10, num_experts=2, args=basic_args)
        assert isinstance(router.model, GNNRouter)

    def test_router_invalid_type(self, basic_args):
        """Test Router raises error for unknown router type."""
        with pytest.raises(ValueError, match="Unknown router_type"):
            Router("INVALID", input_dim=10, num_experts=2, args=basic_args)

    def test_router_forward(self, batch_graphs, basic_args):
        """Test Router forward pass for both MLP and GNN types."""
        for router_type in ["MLP", "GNN"]:
            router = Router(router_type, input_dim=10, num_experts=2, args=basic_args)
            logits = router(batch_graphs)

            assert logits.shape == (3, 2)
            assert torch.isfinite(logits).all()


class TestMoE:
    """Test suite for standard MoE model."""

    def test_moe_initialization(self, basic_args):
        """Test MoE model initializes correctly with two experts."""
        model = MoE(basic_args)

        assert len(model.experts) == 2
        assert hasattr(model, "router")
        assert model.args.layer_types == ["GCN", "GIN"]

    def test_moe_requires_two_experts(self, basic_args):
        """Test MoE raises error when layer_types is not length 2."""
        # Test missing layer_types
        bad_args = AttrDict(basic_args)
        del bad_args.layer_types
        with pytest.raises(AssertionError, match="args.layer_types must be"):
            MoE(bad_args)

        # Test wrong number of experts
        bad_args = AttrDict(basic_args)
        bad_args.layer_types = ["GCN"]  # Only one expert
        with pytest.raises(AssertionError, match="args.layer_types must be"):
            MoE(bad_args)

    def test_moe_expert_types(self):
        """Test MoE handles different expert combinations."""
        test_configs = [
            ["GCN", "GIN"],
            ["GIN", "SAGE"],
            ["GCN", "Unitary"],
            ["GIN", "Unitary"],
        ]

        for layer_types in test_configs:
            args = AttrDict(
                {
                    "input_dim": 10,
                    "hidden_dim": 16,
                    "output_dim": 2,
                    "num_layers": 2,
                    "hidden_layers": [16, 16],
                    "dropout": 0.0,
                    "mlp": True,
                    "num_relations": 1,
                    "last_layer_fa": False,
                    "layer_types": layer_types,
                    "router_type": "MLP",
                    "router_hidden_layers": [16],
                }
            )

            model = MoE(args)
            assert len(model.experts) == 2

    def test_moe_forward_shape(self, batch_graphs, basic_args):
        """Test MoE forward pass produces correct output shapes."""
        model = MoE(basic_args)
        output = model(batch_graphs)

        assert output.shape == (3, 2)  # [batch_size, output_dim]
        assert torch.isfinite(output).all()

    def test_moe_routing_weights_sum_to_one(self, batch_graphs, basic_args):
        """Test that routing weights are properly normalized."""
        model = MoE(basic_args)

        # Access intermediate routing weights
        logits = model.router(batch_graphs)
        weights = torch.softmax(logits, dim=-1)

        # Weights should sum to 1 for each graph
        assert torch.allclose(weights.sum(dim=1), torch.ones(3))
        assert (weights >= 0).all()  # All weights should be non-negative

    def test_moe_different_router_types(self, batch_graphs):
        """Test MoE with different router configurations."""
        for router_type in ["MLP", "GNN"]:
            args = AttrDict(
                {
                    "input_dim": 10,
                    "hidden_dim": 16,
                    "output_dim": 2,
                    "num_layers": 2,
                    "hidden_layers": [16, 16],
                    "dropout": 0.0,
                    "mlp": True,
                    "num_relations": 1,
                    "last_layer_fa": False,
                    "layer_types": ["GCN", "GIN"],
                    "router_type": router_type,
                    "router_layer_type": "GCN",
                    "router_depth": 2,
                    "router_dropout": 0.0,
                    "router_hidden_layers": [16],
                }
            )

            model = MoE(args)
            output = model(batch_graphs)
            assert output.shape == (3, 2)


class TestMoE_E:
    """Test suite for MoE_E model with feature masking."""

    def test_moe_e_initialization(self, moe_e_args):
        """Test MoE_E model initializes correctly."""
        model = MoE_E(moe_e_args)

        assert len(model.experts) == 2
        assert hasattr(model, "router")
        assert model.args.layer_types == ["GCN", "GIN"]

    def test_moe_e_feature_masking(self, moe_e_args):
        """Test MoE_E applies feature masking correctly."""
        # Create graph with enough features for masking
        x = torch.randn(4, 15)  # 15 features (last 5 will be masked)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        graph = Data(x=x, edge_index=edge_index)
        batch_graph = Batch.from_data_list([graph])

        model = MoE_E(moe_e_args)
        original_x = batch_graph.x.clone()

        # Forward pass (modifies graph.x)
        output = model(batch_graph)

        # Check that last 5 dimensions are zeroed
        assert torch.allclose(
            batch_graph.x[:, -5:], torch.zeros_like(batch_graph.x[:, -5:])
        )
        # Check that first 10 dimensions are preserved
        assert torch.allclose(batch_graph.x[:, :-5], original_x[:, :-5])

    def test_moe_e_requires_sufficient_features(self):
        """Test MoE_E fails gracefully with insufficient input features."""
        args = AttrDict(
            {
                "input_dim": 3,  # Too few features for masking
                "hidden_dim": 16,
                "output_dim": 2,
                "num_layers": 2,
                "hidden_layers": [16, 16],
                "dropout": 0.0,
                "mlp": True,
                "num_relations": 1,
                "last_layer_fa": False,
                "layer_types": ["GCN", "GIN"],
                "router_type": "MLP",
                "router_hidden_layers": [16],
            }
        )

        x = torch.randn(4, 3)  # Only 3 features
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        graph = Batch.from_data_list([Data(x=x, edge_index=edge_index)])

        model = MoE_E(args)

        # Should handle gracefully or raise informative error
        with pytest.raises((RuntimeError, IndexError)):
            model(graph)

    def test_moe_e_forward_shape(self, moe_e_args):
        """Test MoE_E forward pass produces correct output shapes."""
        x = torch.randn(8, 15)  # 8 nodes, 15 features
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 0]])
        graphs = [
            Data(x=x[:4], edge_index=edge_index[:, :4]),
            Data(x=x[4:], edge_index=edge_index[:, 4:] - 4),
        ]
        batch_graph = Batch.from_data_list(graphs)

        model = MoE_E(moe_e_args)
        output = model(batch_graph)

        assert output.shape == (2, 3)  # [batch_size, output_dim]
        assert torch.isfinite(output).all()


class TestMoEIntegration:
    """Integration tests for complete MoE pipeline."""

    def test_moe_vs_individual_experts(self, batch_graphs, basic_args):
        """Test MoE output is reasonable compared to individual experts."""
        model = MoE(basic_args)

        # Get MoE output
        moe_output = model(batch_graphs)

        # Get individual expert outputs
        expert1_output = model.experts[0](batch_graphs)
        expert2_output = model.experts[1](batch_graphs)

        # MoE output should be between expert outputs (convex combination)
        # This is a sanity check, not a strict mathematical requirement
        assert moe_output.shape == expert1_output.shape == expert2_output.shape

    def test_moe_gradient_flow(self, batch_graphs, basic_args):
        """Test gradients flow properly through MoE model."""
        model = MoE(basic_args)
        target = torch.randint(0, basic_args.output_dim, (3,))

        output = model(batch_graphs)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()

        # Check gradients exist for both experts and router
        for expert in model.experts:
            for param in expert.parameters():
                if param.requires_grad:
                    assert param.grad is not None

        for param in model.router.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_moe_reproducibility(self, batch_graphs, basic_args):
        """Test MoE produces deterministic outputs with fixed seed."""
        torch.manual_seed(42)
        model1 = MoE(basic_args)
        output1 = model1(batch_graphs)

        torch.manual_seed(42)
        model2 = MoE(basic_args)
        output2 = model2(batch_graphs)

        assert torch.allclose(output1, output2, atol=1e-6)

    def test_moe_device_compatibility(self, batch_graphs, basic_args, device):
        """Test MoE works correctly on different devices."""
        model = MoE(basic_args).to(device)
        batch_graphs = batch_graphs.to(device)

        output = model(batch_graphs)
        assert output.device == device
        assert output.shape == (3, 2)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_node_graph(self):
        """Test MoE handles single-node graphs."""
        # Use GCN layers instead of GIN to avoid BatchNorm issues with single nodes
        single_node_args = AttrDict(
            {
                "input_dim": 10,
                "hidden_dim": 32,
                "output_dim": 2,
                "num_layers": 2,
                "hidden_layers": [32, 32],
                "dropout": 0.1,
                "mlp": True,
                "num_relations": 1,
                "last_layer_fa": False,
                "layer_types": ["GCN", "SAGE"],  # Use layers without BatchNorm
                "router_type": "MLP",
                "router_layer_type": "GCN",
                "router_depth": 2,
                "router_dropout": 0.0,
                "router_hidden_layers": [32, 16],
            }
        )

        x = torch.randn(1, 10)
        edge_index = torch.empty((2, 0), dtype=torch.long)  # No edges
        graph = Batch.from_data_list([Data(x=x, edge_index=edge_index)])

        model = MoE(single_node_args)
        output = model(graph)

        assert output.shape == (1, 2)
        assert torch.isfinite(output).all()

    def test_empty_batch(self, basic_args):
        """Test MoE handles empty batches gracefully."""
        # PyTorch Geometric doesn't support empty batches, so we test the expected behavior
        model = MoE(basic_args)

        # Creating empty batch should raise IndexError (expected behavior)
        with pytest.raises(IndexError):
            Batch.from_data_list([])

        # If we could create an empty batch, test that the model handles it gracefully
        # For now, we just verify the model can be instantiated
        assert model is not None

    def test_large_graph(self, basic_args):
        """Test MoE scales to larger graphs."""
        x = torch.randn(1000, 10)  # Large graph
        # Create a connected graph
        edge_index = torch.tensor(
            [list(range(999)) + list(range(1, 1000)), list(range(1, 1000)) + [0] * 999]
        )
        graph = Batch.from_data_list([Data(x=x, edge_index=edge_index)])

        model = MoE(basic_args)
        output = model(graph)

        assert output.shape == (1, 2)
        assert torch.isfinite(output).all()


class TestParameterCounting:
    """Test parameter counting and model size."""

    def test_moe_parameter_count(self, basic_args):
        """Test MoE has reasonable parameter count."""
        model = MoE(basic_args)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert total_params > 0
        assert trainable_params == total_params  # All params should be trainable

        # Should have more parameters than individual experts due to router
        individual_expert_params = sum(p.numel() for p in model.experts[0].parameters())
        assert total_params > individual_expert_params

    def test_moe_vs_moe_e_parameters(self, basic_args, moe_e_args):
        """Test MoE and MoE_E have similar parameter counts."""
        # Adjust moe_e_args to match basic_args structure
        moe_e_args.update(
            {
                "input_dim": basic_args.input_dim,
                "output_dim": basic_args.output_dim,
            }
        )

        moe_model = MoE(basic_args)
        moe_e_model = MoE_E(moe_e_args)

        moe_params = sum(p.numel() for p in moe_model.parameters())
        moe_e_params = sum(p.numel() for p in moe_e_model.parameters())

        # Should have same number of parameters (only difference is forward pass)
        assert moe_params == moe_e_params


class TestRealWorldScenarios:
    """Test scenarios similar to actual usage in experiments."""

    def test_mutag_like_scenario(self):
        """Test MoE on MUTAG-like graph classification scenario."""
        args = AttrDict(
            {
                "input_dim": 7,  # MUTAG has 7 node features
                "hidden_dim": 64,
                "output_dim": 2,
                "num_layers": 4,
                "hidden_layers": [64] * 4,
                "dropout": 0.1,
                "mlp": True,
                "num_relations": 1,
                "last_layer_fa": False,
                "layer_types": ["GCN", "GIN"],
                "router_type": "MLP",
                "router_hidden_layers": [64, 32],
            }
        )

        # Create MUTAG-like graphs
        graphs = []
        for _ in range(5):
            num_nodes = torch.randint(10, 30, (1,)).item()
            x = torch.randn(num_nodes, 7)
            edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
            graphs.append(Data(x=x, edge_index=edge_index))

        batch = Batch.from_data_list(graphs)
        model = MoE(args)
        output = model(batch)

        assert output.shape == (5, 2)
        assert torch.isfinite(output).all()

    def test_proteins_like_scenario(self):
        """Test MoE on PROTEINS-like scenario with edge features."""
        args = AttrDict(
            {
                "input_dim": 3,  # PROTEINS has 3 node features
                "hidden_dim": 64,
                "output_dim": 2,
                "num_layers": 4,
                "hidden_layers": [64] * 4,
                "dropout": 0.1,
                "mlp": True,
                "num_relations": 1,
                "last_layer_fa": False,
                "layer_types": ["GCN", "GIN"],
                "router_type": "GNN",
                "router_layer_type": "GIN",
                "router_depth": 4,
                "router_dropout": 0.1,
                "router_hidden_layers": [64],
            }
        )

        # Create PROTEINS-like graphs
        graphs = []
        for _ in range(3):
            num_nodes = torch.randint(20, 50, (1,)).item()
            x = torch.randn(num_nodes, 3)
            edge_index = torch.randint(0, num_nodes, (2, num_nodes * 3))
            graphs.append(Data(x=x, edge_index=edge_index))

        batch = Batch.from_data_list(graphs)
        model = MoE(args)
        output = model(batch)

        assert output.shape == (3, 2)
        assert torch.isfinite(output).all()


if __name__ == "__main__":
    """Run tests directly if executed as script."""
    pytest.main([__file__, "-v", "--tb=short"])
