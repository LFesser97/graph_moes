"""Performs the training of the graph classification tasks."""

import copy
import random
from typing import Dict, List, Optional, Tuple

import torch

try:
    from attrdict3 import AttrDict  # Python 3.10+ compatible
except ImportError:
    from attrdict import AttrDict  # Fallback for older Python

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, Subset, random_split
from torch_geometric.loader import DataLoader
from torcheval.metrics import MultilabelAUPRC  # For multi-label datasets like molpcba
from tqdm import tqdm

import wandb
from graph_moes.architectures.graph_model import GNN, GPS, OrthogonalGCN, UnitaryGCN
from graph_moes.moes.graph_moe import MoE, MoE_E
from graph_moes.routing_encodings.encoding_moe import EncodingMoE
from graph_moes.routing_encodings.helper import (
    _create_encoding_moe_loaders,
    _create_encoding_moe_loaders_for_split,
    _get_encoding_moe_encoded_graphs_for_batch,
    _initialize_encoding_moe_model,
)

default_args = AttrDict(
    {
        "learning_rate": 1e-3,
        "max_epochs": 1000000,
        "display": True,
        "device": None,
        "eval_every": 1,
        "stopping_criterion": "validation",
        "stopping_threshold": 1,
        "patience": 50,
        "train_fraction": 0.5,
        "validation_fraction": 0.25,
        "test_fraction": 0.25,
        "dropout": 0.0,
        "weight_decay": 1e-5,
        "input_dim": None,
        "hidden_dim": 64,
        "output_dim": 1,
        "hidden_layers": None,
        "num_layers": 4,
        "batch_size": 64,
        "layer_type": "R-GCN",
        "num_relations": 2,
        "last_layer_fa": False,
        "layer_types": None,
        "router_type": "MLP",
        "router_layer_type": "GIN",
        "router_depth": 4,
        "router_dropout": 0.1,
        "skip_connection": False,  # Whether to use skip/residual connections (for GCN, GIN, SAGE)
        # WandB defaults
        "wandb_enabled": False,
        "wandb_project": "MOE_4",
        "wandb_entity": "weber-geoml-harvard-university",
        "wandb_name": None,
        "wandb_dir": "./wandb",
        "wandb_tags": None,
    }
)


class Experiment:
    def __init__(
        self,
        args: Optional[AttrDict] = None,
        dataset: Optional[Dataset] = None,
        train_dataset: Optional[Dataset] = None,
        validation_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        encoding_moe_encoded_datasets: Optional[Dict[str, Dict[str, List]]] = None,
    ) -> None:
        self.args = default_args + args
        self.dataset = dataset
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.encoding_moe_encoded_datasets = encoding_moe_encoded_datasets
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.categories = None
        self.wandb_active = wandb.run is not None

        if self.args.device is None:
            self.args.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        if self.args.hidden_layers is None:
            self.args.hidden_layers = [self.args.hidden_dim] * self.args.num_layers
        if self.args.input_dim is None:
            try:
                self.args.input_dim = self.dataset[0].x.shape[1]
            except (AttributeError, IndexError):
                self.args.input_dim = 9  # peptides-func
        for graph in self.dataset:
            if not hasattr(graph, "edge_type"):
                num_edges = graph.edge_index.shape[1]
                graph.edge_type = torch.zeros(num_edges, dtype=int)
        if self.args.num_relations is None:
            if self.args.rewiring == "None":
                self.args.num_relations = 1
            else:
                self.args.num_relations = 2

        # Check if EncodingMoE should be used
        is_encoding_moe = (
            hasattr(self.args, "encoding_moe_encodings")
            and self.args.encoding_moe_encodings is not None
            and len(self.args.encoding_moe_encodings) > 0
        )

        if is_encoding_moe:
            # EncodingMoE: Initialize with base dataset dimensions + encoding configs
            base_input_dim = self.args.input_dim
            self.model, self.is_encoding_moe = _initialize_encoding_moe_model(
                self.args,
                base_input_dim,
                self.dataset,
                self.encoding_moe_encoded_datasets,
            )
        elif self.args.layer_type == "GPS":
            self.model = GPS(self.args).to(self.args.device)
            self.is_encoding_moe = False
        elif self.args.layer_type == "Orthogonal":
            self.model = OrthogonalGCN(self.args).to(self.args.device)
            self.is_encoding_moe = False
        elif self.args.layer_type == "Unitary":
            self.model = UnitaryGCN(self.args).to(self.args.device)
            self.is_encoding_moe = False
        elif self.args.layer_type == "MoE":
            self.model = MoE(self.args).to(self.args.device)
            self.is_encoding_moe = False
        elif self.args.layer_type == "MoE_E":
            self.model = MoE_E(self.args).to(self.args.device)
            self.is_encoding_moe = False
        else:
            self.model = GNN(self.args).to(self.args.device)
            self.is_encoding_moe = False

        if self.test_dataset is None:
            dataset_size = len(self.dataset)
            train_size = int(self.args.train_fraction * dataset_size)
            validation_size = int(self.args.validation_fraction * dataset_size)
            # self.train_dataset, self.validation_dataset, self.test_dataset = random_split(self.dataset,[train_size, validation_size, test_size])
            (
                self.train_dataset,
                self.validation_dataset,
                self.test_dataset,
                self.categories,
            ) = custom_random_split(
                self.dataset,
                [
                    self.args.train_fraction,
                    self.args.validation_fraction,
                    self.args.test_fraction,
                ],
            )
        elif self.validation_dataset is None:
            print("self.validation_dataset is None. Custom split will not be used.")
            train_size = int(self.args.train_fraction * len(self.train_dataset))
            validation_size = len(self.args.train_data) - train_size
            self.args.train_data, self.args.validation_data = random_split(
                self.args.train_data, [train_size, validation_size]
            )

    def run(self) -> Tuple[float, float, float, float, Dict[int, int], List[int]]:
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate
        )
        scheduler = ReduceLROnPlateau(optimizer)

        best_validation_acc = 0.0
        best_train_acc = 0.0
        best_test_acc = 0.0
        train_goal = 0.0
        validation_goal = 0.0
        epochs_no_improve = 0
        best_model = copy.deepcopy(self.model)

        train_loader = DataLoader(
            self.train_dataset, batch_size=self.args.batch_size, shuffle=True
        )
        validation_loader = DataLoader(
            self.validation_dataset, batch_size=self.args.batch_size, shuffle=True
        )
        test_loader = DataLoader(
            self.test_dataset, batch_size=self.args.batch_size, shuffle=True
        )
        # complete_loader = DataLoader(self.dataset, batch_size=self.args.batch_size, shuffle=True)
        complete_loader = DataLoader(self.dataset, batch_size=1)

        # create a dictionary of the graphs in the dataset with the key being the graph index
        graph_dict = {}
        for i in range(len(self.dataset)):
            graph_dict[i] = -1

        epoch_pbar = tqdm(range(self.args.max_epochs), desc="Training")
        for epoch in epoch_pbar:
            self.model.train()
            total_loss = 0
            optimizer.zero_grad()

            # Track routing weights for MoE models
            # Note: weights have shape [batch_size, num_experts] - each graph gets its own weights
            routing_weights = []
            is_moe_model = isinstance(self.model, (MoE, MoE_E))

            # For EncodingMoE, we need to load encoded datasets and create batches
            if self.is_encoding_moe:
                dataset_name = getattr(self.args, "dataset", None)
                train_indices = (
                    self.categories[0] if self.categories is not None else None
                )
                encoded_loaders = _create_encoding_moe_loaders(
                    self.args,
                    self.train_dataset,
                    train_indices,
                    self.encoding_moe_encoded_datasets,
                    dataset_name,
                    shuffle=True,
                )
                # Create iterator for encoded loaders
                encoded_iterators = {
                    name: iter(loader) for name, loader in encoded_loaders.items()
                }

            for batch_idx, base_graph in enumerate(train_loader):
                base_graph = base_graph.to(self.args.device)
                # y = base_graph.y.to(self.args.device)

                # for OGB
                y = base_graph.y.flatten()
                y.to(self.args.device)

                # Handle EncodingMoE forward pass
                if self.is_encoding_moe:
                    # Get encoded graphs for this batch
                    encoded_graphs = {}
                    for encoding_name in self.args.encoding_moe_encodings:
                        encoded_batch = _get_encoding_moe_encoded_graphs_for_batch(
                            encoding_name,
                            encoded_iterators,
                            encoded_loaders,
                            self.args.device,
                        )
                        if encoded_batch is not None:
                            encoded_graphs[encoding_name] = encoded_batch

                    # Forward pass through EncodingMoE
                    out, weights = self.model(
                        base_graph, encoded_graphs, return_weights=True
                    )
                    routing_weights.append(weights.detach().cpu())
                # Get routing weights if MoE model
                # weights shape: [batch_size, num_experts] where each row sums to 1.0
                # Example: weights[0] = [0.7, 0.3] means first graph uses 70% expert 0, 30% expert 1
                elif is_moe_model:
                    out, weights = self.model(base_graph, return_weights=True)
                    routing_weights.append(weights.detach().cpu())
                else:
                    out = self.model(base_graph)

                loss = self.loss_fn(input=out, target=y)
                total_loss += loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            scheduler.step(total_loss)

            # Log training loss to wandb
            if self.wandb_active:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train/loss": total_loss.item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                    }
                )

            if epoch % self.args.eval_every == 0:
                if self.args.output_dim == 10:  # peptides-func
                    train_acc = self.test(loader=train_loader)
                    validation_acc = self.test(loader=validation_loader)
                    test_acc = self.test(loader=test_loader)
                else:
                    train_acc = self.eval(loader=train_loader)
                    validation_acc = self.eval(loader=validation_loader)
                    test_acc = self.eval(loader=test_loader)

                # Log accuracies to wandb
                if self.wandb_active:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "train/acc": train_acc,
                            "val/acc": validation_acc,
                            "test/acc": test_acc,
                        }
                    )

                if self.args.stopping_criterion == "train":
                    if train_acc > train_goal:
                        best_train_acc = train_acc
                        best_validation_acc = validation_acc
                        best_test_acc = test_acc
                        epochs_no_improve = 0
                        train_goal = train_acc * self.args.stopping_threshold

                        # Log new best to wandb
                        if self.wandb_active:
                            wandb.log(
                                {
                                    "best/train_acc": best_train_acc,
                                    "best/val_acc": best_validation_acc,
                                    "best/test_acc": best_test_acc,
                                    "best/epoch": epoch,
                                }
                            )
                    elif train_acc > best_train_acc:
                        best_train_acc = train_acc
                        best_validation_acc = validation_acc
                        best_test_acc = test_acc
                        epochs_no_improve += 1
                    else:
                        epochs_no_improve += 1
                elif self.args.stopping_criterion == "validation":
                    if validation_acc > validation_goal:
                        best_train_acc = train_acc
                        best_validation_acc = validation_acc
                        best_test_acc = test_acc
                        epochs_no_improve = 0
                        validation_goal = validation_acc * self.args.stopping_threshold
                        best_model = copy.deepcopy(self.model)

                        # Log new best to wandb
                        if self.wandb_active:
                            wandb.log(
                                {
                                    "best/train_acc": best_train_acc,
                                    "best/val_acc": best_validation_acc,
                                    "best/test_acc": best_test_acc,
                                    "best/epoch": epoch,
                                }
                            )
                    elif validation_acc > best_validation_acc:
                        best_train_acc = train_acc
                        best_validation_acc = validation_acc
                        best_test_acc = test_acc
                        epochs_no_improve += 1
                    else:
                        epochs_no_improve += 1
                if self.args.display:
                    epoch_pbar.set_postfix(
                        {
                            "Train": f"{train_acc:.3f}",
                            "Val": f"{validation_acc:.3f}",
                            "Test": f"{test_acc:.3f}",
                            "Best": f"{best_validation_acc:.3f}",
                        }
                    )
                if epochs_no_improve > self.args.patience:
                    if self.args.display:
                        print(
                            f"{self.args.patience} epochs without improvement, stopping training"
                        )
                        print(
                            f"Best train acc: {best_train_acc}, Best validation acc: {best_validation_acc}, Best test acc: {best_test_acc}"
                        )

                        # Log final results to wandb
                        if self.wandb_active:
                            wandb.log(
                                {
                                    "final/train_acc": best_train_acc,
                                    "final/val_acc": best_validation_acc,
                                    "final/test_acc": best_test_acc,
                                    "final/epoch": epoch,
                                    "final/early_stopped": True,
                                }
                            )

                        energy = 0

                        # evaluate the model on all graphs in the dataset
                        # and record the correctness for each graph in the test set
                        assert (
                            best_model != self.model
                        ), "Best model is the same as the current model"

                        # Handle EncodingMoE for best_model evaluation
                        if self.is_encoding_moe:
                            dataset_name = getattr(self.args, "dataset", None)
                            test_indices = self.categories[2] if self.categories else []

                            # Create encoded loaders for test set
                            encoded_loaders_test = {}
                            if self.encoding_moe_encoded_datasets and dataset_name:
                                for encoding_name in self.args.encoding_moe_encodings:
                                    if (
                                        encoding_name
                                        in self.encoding_moe_encoded_datasets
                                        and dataset_name
                                        in self.encoding_moe_encoded_datasets[
                                            encoding_name
                                        ]
                                    ):
                                        encoded_dataset = (
                                            self.encoding_moe_encoded_datasets[
                                                encoding_name
                                            ][dataset_name]
                                        )
                                        encoded_test = [
                                            encoded_dataset[i]
                                            for i in test_indices
                                            if i < len(encoded_dataset)
                                        ]
                                        from torch_geometric.loader import (
                                            DataLoader as PyGDataLoader,
                                        )

                                        encoded_loaders_test[encoding_name] = (
                                            PyGDataLoader(
                                                encoded_test,
                                                batch_size=1,  # One graph at a time
                                                shuffle=False,
                                            )
                                        )

                            encoded_iterators_test = {
                                name: iter(loader)
                                for name, loader in encoded_loaders_test.items()
                            }

                            for graph, i in zip(
                                complete_loader, range(len(self.dataset))
                            ):
                                if (
                                    i in self.categories[2]
                                ):  # Only track test set graphs
                                    graph = graph.to(self.args.device)
                                    y = graph.y.to(self.args.device)

                                    # Get encoded graphs for this single graph
                                    encoded_graphs = {}
                                    for (
                                        encoding_name
                                    ) in self.args.encoding_moe_encodings:
                                        if encoding_name in encoded_iterators_test:
                                            try:
                                                encoded_graph = next(
                                                    encoded_iterators_test[
                                                        encoding_name
                                                    ]
                                                )
                                                encoded_graph = encoded_graph.to(
                                                    self.args.device
                                                )
                                                encoded_graphs[encoding_name] = (
                                                    encoded_graph
                                                )
                                            except StopIteration:
                                                pass

                                    out = best_model(graph, encoded_graphs)
                                    _, pred = out.max(dim=1)
                                    graph_dict[i] = pred.eq(y).sum().item()
                        else:
                            for graph, i in zip(
                                complete_loader, range(len(self.dataset))
                            ):
                                if (
                                    i in self.categories[2]
                                ):  # Only track test set graphs
                                    graph = graph.to(self.args.device)
                                    y = graph.y.to(self.args.device)
                                    out = best_model(graph)
                                    _, pred = out.max(dim=1)
                                    graph_dict[i] = pred.eq(y).sum().item()
                        print("Computed correctness for each graph in the test dataset")

                        # save the model
                        # torch.save(best_model.state_dict(), "model.pth")

                        # get the current directory and print it
                        # print("Saved model in directory: ", os.getcwd())

                    return (
                        best_train_acc,
                        best_validation_acc,
                        best_test_acc,
                        energy,
                        graph_dict,
                        self.categories[2],  # Return test set indices
                    )

        if self.args.display:
            print("Reached max epoch count, stopping training")
            print(
                f"Best train acc: {best_train_acc}, Best validation acc: {best_validation_acc}, Best test acc: {best_test_acc}"
            )

        # Log final results to wandb
        if self.wandb_active:
            wandb.log(
                {
                    "final/train_acc": best_train_acc,
                    "final/val_acc": best_validation_acc,
                    "final/test_acc": best_test_acc,
                    "final/epoch": self.args.max_epochs,
                    "final/early_stopped": False,
                }
            )

        energy = 0
        # If we reach max epochs, still evaluate test set graphs
        if hasattr(self, "categories") and self.categories is not None:
            # Handle EncodingMoE for final model evaluation
            if self.is_encoding_moe:
                dataset_name = getattr(self.args, "dataset", None)
                test_indices = self.categories[2]

                # Create encoded loaders for test set
                encoded_loaders_test = {}
                if self.encoding_moe_encoded_datasets and dataset_name:
                    for encoding_name in self.args.encoding_moe_encodings:
                        if (
                            encoding_name in self.encoding_moe_encoded_datasets
                            and dataset_name
                            in self.encoding_moe_encoded_datasets[encoding_name]
                        ):
                            encoded_dataset = self.encoding_moe_encoded_datasets[
                                encoding_name
                            ][dataset_name]
                            encoded_test = [
                                encoded_dataset[i]
                                for i in test_indices
                                if i < len(encoded_dataset)
                            ]
                            from torch_geometric.loader import (
                                DataLoader as PyGDataLoader,
                            )

                            encoded_loaders_test[encoding_name] = PyGDataLoader(
                                encoded_test,
                                batch_size=1,  # One graph at a time
                                shuffle=False,
                            )

                encoded_iterators_test = {
                    name: iter(loader) for name, loader in encoded_loaders_test.items()
                }

                for graph, i in zip(complete_loader, range(len(self.dataset))):
                    if i in self.categories[2]:  # Only track test set graphs
                        graph = graph.to(self.args.device)
                        y = graph.y.to(self.args.device)

                        # Get encoded graphs for this single graph
                        encoded_graphs = {}
                        for encoding_name in self.args.encoding_moe_encodings:
                            if encoding_name in encoded_iterators_test:
                                try:
                                    encoded_graph = next(
                                        encoded_iterators_test[encoding_name]
                                    )
                                    encoded_graph = encoded_graph.to(self.args.device)
                                    encoded_graphs[encoding_name] = encoded_graph
                                except StopIteration:
                                    pass

                        out = self.model(graph, encoded_graphs)
                        _, pred = out.max(dim=1)
                        graph_dict[i] = pred.eq(y).sum().item()
            else:
                for graph, i in zip(complete_loader, range(len(self.dataset))):
                    if i in self.categories[2]:  # Only track test set graphs
                        graph = graph.to(self.args.device)
                        y = graph.y.to(self.args.device)
                        out = self.model(graph)
                        _, pred = out.max(dim=1)
                        graph_dict[i] = pred.eq(y).sum().item()
            test_indices = self.categories[2]
        else:
            test_indices = []
        return (
            best_train_acc,
            best_validation_acc,
            best_test_acc,
            energy,
            graph_dict,
            test_indices,
        )

    def eval(self, loader: DataLoader) -> float:
        self.model.eval()
        sample_size = len(loader.dataset)

        # Handle EncodingMoE - need encoded graphs
        if self.is_encoding_moe:
            dataset_name = getattr(self.args, "dataset", None)
            encoded_loaders = _create_encoding_moe_loaders_for_split(
                self.args,
                loader,
                self.train_dataset,
                self.validation_dataset,
                self.categories,
                self.encoding_moe_encoded_datasets,
                dataset_name,
            )
            # Create iterators for encoded loaders
            encoded_iterators = {
                name: iter(loader) for name, loader in encoded_loaders.items()
            }

            with torch.no_grad():
                total_correct = 0
                for base_graph in loader:
                    base_graph = base_graph.to(self.args.device)

                    # Get encoded graphs for this batch
                    encoded_graphs = {}
                    for encoding_name in self.args.encoding_moe_encodings:
                        encoded_batch = _get_encoding_moe_encoded_graphs_for_batch(
                            encoding_name,
                            encoded_iterators,
                            encoded_loaders,
                            self.args.device,
                        )
                        if encoded_batch is not None:
                            encoded_graphs[encoding_name] = encoded_batch

                    # Forward pass through EncodingMoE
                    out = self.model(base_graph, encoded_graphs)
                    y = base_graph.y.flatten().to(self.args.device)

                    # check if y contains more than one element
                    if y.dim() > 1:
                        loss = self.loss_fn(input=out, target=y)
                        total_correct -= loss
                    else:
                        _, pred = out.max(dim=1)
                        total_correct += pred.eq(y).sum().item()
        else:
            # Regular model - standard eval
            with torch.no_grad():
                total_correct = 0
                for graph in loader:
                    graph = graph.to(self.args.device)
                    out = self.model(graph)
                    y = graph.y.to(self.args.device)
                    # check if y contains more than one element
                    if y.dim() > 1:
                        loss = self.loss_fn(input=out, target=y)
                        total_correct -= loss
                    else:
                        _, pred = out.max(dim=1)
                        total_correct += pred.eq(y).sum().item()

        return total_correct / sample_size

    @torch.no_grad()
    def test(self, loader: DataLoader) -> float:
        self.model.eval()
        sample_size = len(loader.dataset)

        # Choose metric based on dataset type
        # Multi-label datasets need AUPRC, multi-class datasets need accuracy
        dataset_name = getattr(self.args, "dataset", "")
        multi_label_datasets = ["molpcba"]  # Known multi-label datasets
        use_auprc = (
            dataset_name in multi_label_datasets or self.args.output_dim > 50
        )  # Fallback for high-dim outputs

        # Handle EncodingMoE - need encoded graphs
        if self.is_encoding_moe:
            dataset_name = getattr(self.args, "dataset", None)
            encoded_loaders = _create_encoding_moe_loaders_for_split(
                self.args,
                loader,
                self.train_dataset,
                self.validation_dataset,
                self.categories,
                self.encoding_moe_encoded_datasets,
                dataset_name,
            )
            # Create iterators for encoded loaders
            encoded_iterators = {
                name: iter(loader) for name, loader in encoded_loaders.items()
            }

            if use_auprc:
                # Use AUPRC for multi-label datasets (like molpcba with 128 classes)
                metric = MultilabelAUPRC(num_labels=self.args.output_dim)

                for base_graph in loader:
                    base_graph = base_graph.to(self.args.device)

                    # Get encoded graphs for this batch
                    encoded_graphs = {}
                    for encoding_name in self.args.encoding_moe_encodings:
                        encoded_batch = _get_encoding_moe_encoded_graphs_for_batch(
                            encoding_name,
                            encoded_iterators,
                            encoded_loaders,
                            self.args.device,
                        )
                        if encoded_batch is not None:
                            encoded_graphs[encoding_name] = encoded_batch

                    # Forward pass through EncodingMoE
                    out = self.model(base_graph, encoded_graphs)
                    y = base_graph.y.flatten().to(self.args.device)

                    # For multi-label, convert single labels to multi-hot if needed
                    if y.dim() == 1:
                        # Convert to multi-hot encoding for AUPRC
                        y_multihot = torch.zeros(
                            out.shape[0], self.args.output_dim, device=self.args.device
                        )
                        y_multihot.scatter_(1, y.unsqueeze(1), 1)
                        y = y_multihot

                    metric.update(out, y)

                return metric.compute().item()
            else:
                # Use accuracy for multi-class datasets
                total_correct = 0
                for base_graph in loader:
                    base_graph = base_graph.to(self.args.device)

                    # Get encoded graphs for this batch
                    encoded_graphs = {}
                    for encoding_name in self.args.encoding_moe_encodings:
                        encoded_batch = _get_encoding_moe_encoded_graphs_for_batch(
                            encoding_name,
                            encoded_iterators,
                            encoded_loaders,
                            self.args.device,
                        )
                        if encoded_batch is not None:
                            encoded_graphs[encoding_name] = encoded_batch

                    # Forward pass through EncodingMoE
                    out = self.model(base_graph, encoded_graphs)
                    y = base_graph.y.flatten().to(self.args.device)

                    # Handle both multi-class and multi-label cases
                    if y.dim() > 1:
                        # Multi-label case - use sigmoid + threshold
                        pred = (torch.sigmoid(out) > 0.5).float()
                        total_correct += (pred == y).all(dim=1).sum().item()
                    else:
                        # Multi-class case - use argmax
                        _, pred = out.max(dim=1)
                        total_correct += pred.eq(y).sum().item()

                return total_correct / sample_size
        else:
            # Regular model - standard test
            if use_auprc:
                # Use AUPRC for multi-label datasets (like molpcba with 128 classes)
                metric = MultilabelAUPRC(num_labels=self.args.output_dim)

                for data in loader:
                    data = data.to(self.args.device)
                    out = self.model(data)
                    y = data.y.to(self.args.device)

                    # For multi-label, convert single labels to multi-hot if needed
                    if y.dim() == 1:
                        # Convert to multi-hot encoding for AUPRC
                        y_multihot = torch.zeros(
                            out.shape[0], self.args.output_dim, device=self.args.device
                        )
                        y_multihot.scatter_(1, y.unsqueeze(1), 1)
                        y = y_multihot

                    metric.update(out, y)

                return metric.compute().item()
            else:
                # Use accuracy for multi-class datasets
                with torch.no_grad():
                    total_correct = 0
                    for data in loader:
                        data = data.to(self.args.device)
                        out = self.model(data)
                        y = data.y.to(self.args.device)

                        # Handle both multi-class and multi-label cases
                        if y.dim() > 1:
                            # Multi-label case - use sigmoid + threshold
                            pred = (torch.sigmoid(out) > 0.5).float()
                            total_correct += (pred == y).all(dim=1).sum().item()
                        else:
                            # Multi-class case - use argmax
                            _, pred = out.max(dim=1)
                            total_correct += pred.eq(y).sum().item()

                return total_correct / sample_size

    def check_dirichlet(self, loader: DataLoader) -> float:
        self.model.eval()
        sample_size = len(loader.dataset)
        with torch.no_grad():
            total_energy = 0
            for graph in loader:
                graph = graph.to(self.args.device)
                total_energy += self.model(graph, measure_dirichlet=True)
        return total_energy / sample_size


def custom_random_split(
    dataset: Dataset, percentages: List[float]
) -> Tuple[Subset, ...]:
    percentages = [100 * percentage for percentage in percentages]
    if sum(percentages) != 100:
        raise ValueError("Percentages must sum to 100")

    # Calculate the lengths of the three categories
    total_length = len(dataset)
    lengths = [int(total_length * p / 100) for p in percentages]

    # Shuffle the input list
    shuffled_list = [*range(total_length)]
    random.shuffle(shuffled_list)

    # Split the shuffled list into three categories
    categories = [
        shuffled_list[: lengths[0]],
        shuffled_list[lengths[0] : lengths[0] + lengths[1]],
        shuffled_list[lengths[0] + lengths[1] :],
    ]

    train_dataset = Subset(dataset, categories[0])
    validation_dataset = Subset(dataset, categories[1])
    test_dataset = Subset(dataset, categories[2])

    return train_dataset, validation_dataset, test_dataset, categories
