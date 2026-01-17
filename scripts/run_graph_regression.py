"""Script to run graph regression experiments.


This script orchestrates comprehensive graph regression experiments on molecular datasets,
primarily focusing on the ZINC dataset for molecular property prediction tasks. It supports
various graph neural network architectures including GCN, GIN, SAGE, GAT, and Mixture of Experts.

The script handles dataset loading, optional structural encodings (like Laplacian eigenvectors,
random walk positional encodings, and curvature-based features), model training across multiple
trials, and comprehensive result logging with statistical analysis of performance metrics.
"""

import os
import pickle
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T

try:
    from attrdict3 import AttrDict  # Python 3.10+ compatible
except ImportError:
    from attrdict import AttrDict  # Fallback for older Python
from torch_geometric.data import Data
from torch_geometric.datasets import ZINC

import wandb
from graph_moes.download.load_graphbench import load_graphbench_dataset
from graph_moes.encodings.custom_encodings import LocalCurvatureProfile
from graph_moes.experiments.graph_regression import Experiment
from graph_moes.experiments.track_avg_accuracy import load_and_plot_average_per_graph
from hyperparams import get_args_from_input

# import custom encodings


def _convert_lrgb(dataset: torch.Tensor) -> Data:
    """Convert LRGB dataset tuple format to PyTorch Geometric Data object.

    Args:
        dataset: Tuple containing (x, edge_attr, edge_index, y) tensors

    Returns:
        PyTorch Geometric Data object with node features, edges, and labels
    """
    x = dataset[0]
    edge_attr = dataset[1]
    edge_index = dataset[2]
    y = dataset[3]

    return Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)


# load zinc
data_directory = (
    "/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/graph_moes/graph_datasets"
)
os.makedirs(data_directory, exist_ok=True)
train_dataset = ZINC(path=data_directory, subset=True, split="train")
val_dataset = ZINC(path=data_directory, subset=True, split="val")
test_dataset = ZINC(path=data_directory, subset=True, split="test")
zinc = (
    [train_dataset[i] for i in range(len(train_dataset))]
    + [val_dataset[i] for i in range(len(val_dataset))]
    + [test_dataset[i] for i in range(len(test_dataset))]
)


"""
# load peptides
peptides_zip_filepath = os.getcwd()
peptides_train = torch.load(os.path.join(data_directory, "peptidesstruct", "train.pt"))
peptides_val = torch.load(os.path.join(data_directory, "peptidesstruct", "val.pt"))
peptides_test = torch.load(os.path.join(data_directory, "peptidesstruct", "test.pt"))
peptides_struct = [_convert_lrgb(peptides_train[i]) for i in range(len(peptides_train))] + [_convert_lrgb(peptides_val[i]) for i in range(len(peptides_val))] + [_convert_lrgb(peptides_test[i]) for i in range(len(peptides_test))]
"""

# GraphBench datasets (graph regression tasks)
print("\n📊 Loading GraphBench regression datasets...")
graphbench_datasets = {}

# GraphBench dataset names that are relevant for graph regression
# Weather forecasting and some circuit/chip design tasks may be regression
graphbench_regression_datasets = [
    "weather",  # Weather forecasting (regression)
    # Note: Add other regression datasets as they become available/identified
]

for dataset_name in graphbench_regression_datasets:
    try:
        print(f"  ⏳ Loading GraphBench: {dataset_name}...")
        graphbench_data = load_graphbench_dataset(
            dataset_name=dataset_name, root=data_directory
        )
        graphbench_datasets[f"graphbench_{dataset_name}"] = graphbench_data
        print(f"  ✅ GraphBench {dataset_name} loaded: {len(graphbench_data)} graphs")
    except (ImportError, ValueError, RuntimeError, OSError) as e:
        print(
            f"  ⚠️  Failed to load GraphBench {dataset_name}: {e} (may not be installed or available)"
        )

# we can pass them as click args too
# datasets = {"zinc": zinc, "peptides_struct": peptides_struct}
datasets = {"zinc": zinc, **graphbench_datasets}


def log_to_file(message: str, filename: str = "results/graph_regression.txt") -> None:
    """Log a message to both console and file.

    Args:
        message: The message to log
        filename: Path to the log file (default: "results/graph_regression.txt")
    """
    print(message)
    file = open(filename, "a")
    file.write(message)
    file.close()


default_args = AttrDict(
    {
        "dropout": 0.1,
        "num_layers": 16,
        "hidden_dim": 64,
        "learning_rate": 1e-3,
        "layer_type": "GINE",
        "display": True,
        "num_trials": 200,
        "eval_every": 1,
        "patience": 250,
        "output_dim": 2,
        "alpha": 0.1,
        "eps": 0.001,
        "dataset": None,
        "last_layer_fa": False,
        "encoding": None,
        # WandB defaults
        "wandb_enabled": False,
        "wandb_project": "MOE_new",
        "wandb_entity": "weber-geoml-harvard-university",
        "wandb_name": None,
        "wandb_dir": "./wandb",
        "wandb_tags": None,
    }
)

hyperparams = {
    "zinc": AttrDict({"output_dim": 1}),
    "peptides_struct": AttrDict({"output_dim": 11}),
    # GraphBench regression datasets - output_dim may need adjustment
    "graphbench_weather": AttrDict(
        {"output_dim": 1}
    ),  # Placeholder - adjust based on actual task - NEED TO VERIFY WITH THE TEST SHAPE TODO TODO
}


results = []
args = default_args
args += get_args_from_input()
if args.dataset:
    # restricts to just the given dataset if this mode is chosen
    name = args.dataset
    datasets = {name: datasets[name]}

for key in datasets:
    args += hyperparams[key]
    train_accuracies = []
    validation_accuracies = []
    test_accuracies = []
    energies = []
    print(f"TESTING: {key} ({args.encoding} - layer {args.layer_type})")

    dataset = datasets[key]

    # encode the dataset using the given encoding, if args.encoding is not None
    if args.encoding in ["LAPE", "RWPE", "LCP", "LDP", "SUB", "EGO"]:

        if os.path.exists(f"data/{key}_{args.encoding}.pt"):
            print("ENCODING ALREADY COMPLETED...")
            dataset = torch.load(f"data/{key}_{args.encoding}.pt")

        elif args.encoding == "LCP":
            print("ENCODING STARTED...")
            lcp = LocalCurvatureProfile()
            for i in range(len(dataset)):
                dataset[i] = lcp.compute_orc(dataset[i])
                print(f"Graph {i} of {len(dataset)} encoded with {args.encoding}")
            torch.save(dataset, f"data/{key}_{args.encoding}.pt")

        else:
            print("ENCODING STARTED...")
            org_dataset_len = len(dataset)
            drop_datasets = []
            current_graph = 0

            for i in range(org_dataset_len):
                if args.encoding == "LAPE":
                    num_nodes = dataset[i].num_nodes
                    eigvecs = np.min([num_nodes, 8]) - 2
                    transform = T.AddLaplacianEigenvectorPE(k=eigvecs)

                elif args.encoding == "RWPE":
                    transform = T.AddRandomWalkPE(walk_length=16)

                elif args.encoding == "LDP":
                    transform = T.LocalDegreeProfile()

                elif args.encoding == "SUB":
                    transform = T.RootedRWSubgraph(walk_length=10)

                elif args.encoding == "EGO":
                    transform = T.RootedEgoNets(num_hops=2)

                try:
                    dataset[i] = transform(dataset[i])
                    print(
                        f"Graph {current_graph} of {org_dataset_len} encoded with {args.encoding}"
                    )
                    current_graph += 1

                except Exception:
                    print(
                        f"Graph {current_graph} of {org_dataset_len} dropped due to encoding error"
                    )
                    drop_datasets.append(i)
                    current_graph += 1

            for i in sorted(drop_datasets, reverse=True):
                dataset.pop(i)

            # save the dataset to a file in the data folder
            torch.save(dataset, f"data/{key}_{args.encoding}.pt")

    # create a dictionary of the graphs in the dataset with the key being the graph index
    # graph_dict[graph_idx] = list of error values (MAE) for each test appearance
    graph_dict = {}
    for i in range(len(dataset)):
        graph_dict[i] = []

    # Track how many times each graph has appeared in test sets
    test_appearances = {i: 0 for i in range(len(dataset))}
    required_test_appearances = 10  # Each graph should appear in test sets 10 times

    print("GRAPH DICTIONARY CREATED...")
    print(
        f"🎯 Goal: Each graph should appear in test sets {required_test_appearances} times"
    )

    # spectral_gap = average_spectral_gap(dataset)
    print("TRAINING STARTED...")
    start = time.time()

    # Create experiment group for wandb
    experiment_id = None
    if args.wandb_enabled:
        # Create a unique experiment group ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{key}_{args.layer_type}_{timestamp}"
        if args.wandb_name:
            experiment_id = f"{args.wandb_name}_{timestamp}"
        print(f"🔬 WandB Experiment Group: {experiment_id}")

    trial = 0
    while True:
        trial += 1

        # Check if we've satisfied the requirement (before running this trial)
        min_test_appearances = (
            min(test_appearances.values())
            if test_appearances and len(test_appearances) > 0
            else 0
        )
        if min_test_appearances >= required_test_appearances:
            print(
                f"\n✅ All graphs have appeared in test sets at least {required_test_appearances} times!"
            )
            print(f"   Stopping before trial {trial}")
            break

        # Also respect the num_trials limit if set (for backwards compatibility)
        if (
            hasattr(args, "num_trials")
            and args.num_trials is not None
            and trial > args.num_trials
        ):
            print(
                f"\n⚠️  Reached num_trials limit ({args.num_trials}), but not all graphs have appeared 10 times yet"
            )
            print(
                f"   Current min appearances: {min_test_appearances}/{required_test_appearances}"
            )
            break

        print(
            f"\n📊 TRIAL {trial} (Min test appearances before trial: {min_test_appearances}/{required_test_appearances})"
        )

        # Initialize wandb for this specific trial
        if args.wandb_enabled:
            trial_run_name = f"trial_{trial + 1:02d}"
            if args.wandb_name:
                trial_run_name = f"{args.wandb_name}_trial_{trial + 1:02d}"

            # Build config dictionary
            wandb_config = {
                **dict(args),
                "trial_num": trial,
                "dataset": key,
                # Add grouping variables:
                "dataset_name": key,
                "is_moe": hasattr(args, "layer_types") and args.layer_types is not None,
                "model_type": (
                    "MoE"
                    if hasattr(args, "layer_types") and args.layer_types
                    else args.layer_type
                ),
                "num_layers": args.num_layers,
            }
            # Add router configuration ONLY for MoE models
            if hasattr(args, "layer_types") and args.layer_types is not None:
                if hasattr(args, "router_type"):
                    wandb_config["router_type"] = args.router_type
                if hasattr(args, "router_layer_type"):
                    wandb_config["router_layer_type"] = args.router_layer_type
                if hasattr(args, "router_depth"):
                    wandb_config["router_depth"] = args.router_depth
                if hasattr(args, "router_dropout"):
                    wandb_config["router_dropout"] = args.router_dropout

            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=trial_run_name,
                group=experiment_id,  # Group all trials together
                config=wandb_config,
                dir=args.wandb_dir,
                tags=list(args.wandb_tags or []) + [f"trial_{trial}"],
                reinit=True,  # Allow multiple runs in same process
            )
            print(f"🚀 WandB Trial Run: {wandb.run.name}")

        try:
            train_acc, validation_acc, test_acc, energy, dictionary, test_indices = (
                Experiment(args=args, dataset=dataset).run()
            )

            train_accuracies.append(train_acc)
            validation_accuracies.append(validation_acc)
            test_accuracies.append(test_acc)
            energies.append(energy)

            # Track test set appearances and record error only for test set graphs
            for graph_idx in test_indices:
                test_appearances[graph_idx] += 1
                if graph_idx in dictionary and dictionary[graph_idx] != -1:
                    graph_dict[graph_idx].append(dictionary[graph_idx])

            # Show intermediate results
            min_appearances = min(test_appearances.values())
            max_appearances = max(test_appearances.values())
            print(
                f"   Train MAE: {train_acc:.4f} | Val MAE: {validation_acc:.4f} | Test MAE: {test_acc:.4f}"
            )
            print(
                f"   Test appearances: min={min_appearances}, max={max_appearances}, graphs_in_test={len(test_indices)}"
            )

            # Log final trial results to this trial's wandb run
            if args.wandb_enabled:
                log_dict = {
                    "final/train_mae": train_acc,
                    "final/val_mae": validation_acc,
                    "final/test_mae": test_acc,
                    "final/energy": energy,
                    "final/min_test_appearances": min_appearances,
                    "final/max_test_appearances": max_appearances,
                    # Add grouping variables:
                    "groupby/dataset": key,
                    "groupby/model_type": (
                        "MoE"
                        if hasattr(args, "layer_types") and args.layer_types
                        else args.layer_type
                    ),
                    "groupby/is_moe": hasattr(args, "layer_types")
                    and args.layer_types is not None,
                    "groupby/moe_layers": (
                        "+".join(args.layer_types)
                        if hasattr(args, "layer_types") and args.layer_types is not None
                        else None
                    ),
                    "groupby/num_layers": args.num_layers,
                }
                # Add router info ONLY for MoE models
                if hasattr(args, "layer_types") and args.layer_types is not None:
                    if hasattr(args, "router_type"):
                        log_dict["groupby/router_type"] = args.router_type
                    if hasattr(args, "router_layer_type"):
                        log_dict["groupby/router_layer_type"] = args.router_layer_type
                    if hasattr(args, "router_depth"):
                        log_dict["groupby/router_depth"] = args.router_depth
                    if hasattr(args, "router_dropout"):
                        log_dict["groupby/router_dropout"] = args.router_dropout
                wandb.log(log_dict)

        finally:
            # Finish this trial's wandb run
            if args.wandb_enabled:
                wandb.finish()

    end = time.time()
    run_duration = end - start
    print(f"⏱️  Training completed in {run_duration:.2f} seconds")
    print(f"📊 Total trials run: {trial}")
    print(
        f"📈 Final test appearances: min={min(test_appearances.values())}, max={max(test_appearances.values())}"
    )

    # Calculate how many graphs have appeared the required number of times
    graphs_with_sufficient_appearances = sum(
        1 for count in test_appearances.values() if count >= required_test_appearances
    )
    print(
        f"✅ Graphs with ≥{required_test_appearances} test appearances: {graphs_with_sufficient_appearances}/{len(dataset)}"
    )

    # Save graph_dict (per-graph error tracking) and test_appearances
    os.makedirs(f"results/{args.num_layers}_layers", exist_ok=True)
    graph_dict_filename = f"results/{args.num_layers}_layers/{key}_{args.layer_type}_{args.encoding}_graph_dict.pickle"
    with open(graph_dict_filename, "wb") as f:
        pickle.dump(
            {
                "graph_dict": graph_dict,
                "test_appearances": test_appearances,
                "required_test_appearances": required_test_appearances,
            },
            f,
        )
    print(f"💾 Graph error dictionary saved to: {graph_dict_filename}")

    # Generate and save average error per graph plot
    try:
        plot_path = load_and_plot_average_per_graph(
            graph_dict_filename,
            dataset_name=key,
            layer_type=args.layer_type,
            encoding=args.encoding,
            num_layers=args.num_layers,
            task_type="regression",
            output_dir="results",
        )
        if plot_path:
            print(f"📊 Average error plot saved to: {plot_path}")
    except Exception as e:
        print(f"⚠️  Failed to generate average error plot: {e}")

    # Calculate statistics
    num_trials_actual = len(train_accuracies)
    train_mean = np.mean(train_accuracies)
    val_mean = np.mean(validation_accuracies)
    test_mean = np.mean(test_accuracies)
    energy_mean = 100 * np.mean(energies)
    train_ci = (
        2 * np.std(train_accuracies) / (num_trials_actual**0.5)
        if num_trials_actual > 0
        else 0
    )
    val_ci = (
        2 * np.std(validation_accuracies) / (num_trials_actual**0.5)
        if num_trials_actual > 0
        else 0
    )
    test_ci = (
        2 * np.std(test_accuracies) / (num_trials_actual**0.5)
        if num_trials_actual > 0
        else 0
    )
    energy_ci = (
        200 * np.std(energies) / (num_trials_actual**0.5)
        if num_trials_actual > 0
        else 0
    )
    log_to_file(
        f"RESULTS FOR dataset: {key} (model: {args.layer_type}), with encodings: {args.encoding}:\n"
    )
    log_to_file(f"average mae: {test_mean}\n")
    log_to_file(f"plus/minus:  {test_ci}\n\n")

    results.append(
        {
            "dataset": key,
            "encoding": args.encoding,
            "layer_type": args.layer_type,
            "alpha": args.alpha,
            "eps": args.eps,
            "test_mean": test_mean,
            "test_ci": test_ci,
            "val_mean": val_mean,
            "val_ci": val_ci,
            "train_mean": train_mean,
            "train_ci": train_ci,
            "energy_mean": energy_mean,
            "energy_ci": energy_ci,
            "last_layer_fa": args.last_layer_fa,
            "run_duration": run_duration,
            "num_trials_actual": num_trials_actual,
            "min_test_appearances": min(test_appearances.values()),
            "max_test_appearances": max(test_appearances.values()),
            "graphs_with_sufficient_appearances": graphs_with_sufficient_appearances,
        }
    )

    # Create a summary run for the overall experiment
    if args.wandb_enabled:
        summary_run_name = "SUMMARY"
        if args.wandb_name:
            summary_run_name = f"{args.wandb_name}_SUMMARY"

        # Build config dictionary for summary run
        summary_config = {
            **dict(args),
            "dataset": key,
            "run_type": "summary",
            "num_trials": num_trials_actual,
            "required_test_appearances": required_test_appearances,
            # Add grouping variables:
            "groupby/dataset": key,
            "groupby/model_type": (
                "MoE"
                if hasattr(args, "layer_types") and args.layer_types
                else args.layer_type
            ),
            "groupby/is_moe": hasattr(args, "layer_types")
            and args.layer_types is not None,
            "groupby/moe_layers": (
                "+".join(args.layer_types)
                if hasattr(args, "layer_types") and args.layer_types is not None
                else None
            ),
            "groupby/num_layers": args.num_layers,
        }
        # Add router configuration ONLY for MoE models
        if hasattr(args, "layer_types") and args.layer_types is not None:
            if hasattr(args, "router_type"):
                summary_config["router_type"] = args.router_type
            if hasattr(args, "router_layer_type"):
                summary_config["router_layer_type"] = args.router_layer_type
            if hasattr(args, "router_depth"):
                summary_config["router_depth"] = args.router_depth
            if hasattr(args, "router_dropout"):
                summary_config["router_dropout"] = args.router_dropout

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=summary_run_name,
            group=experiment_id,
            config=summary_config,
            dir=args.wandb_dir,
            tags=list(args.wandb_tags or []) + ["summary"],
            reinit=True,
        )

        # Log aggregate statistics
        summary_log_dict = {
            "summary/test_mean": test_mean,
            "summary/test_ci": test_ci,
            "summary/val_mean": val_mean,
            "summary/val_ci": val_ci,
            "summary/train_mean": train_mean,
            "summary/train_ci": train_ci,
            "summary/energy_mean": energy_mean,
            "summary/energy_ci": energy_ci,
            "summary/run_duration": run_duration,
            "summary/num_trials_actual": num_trials_actual,
            "summary/min_test_appearances": min(test_appearances.values()),
            "summary/max_test_appearances": max(test_appearances.values()),
            "summary/graphs_with_sufficient_appearances": graphs_with_sufficient_appearances,
            # Log individual trial results for analysis
            "trials/train_maes": train_accuracies,
            "trials/val_maes": validation_accuracies,
            "trials/test_maes": test_accuracies,
            "trials/energies": energies,
        }
        # Add router info ONLY for MoE models
        if hasattr(args, "layer_types") and args.layer_types is not None:
            if hasattr(args, "router_type"):
                summary_log_dict["groupby/router_type"] = args.router_type
            if hasattr(args, "router_layer_type"):
                summary_log_dict["groupby/router_layer_type"] = args.router_layer_type
            if hasattr(args, "router_depth"):
                summary_log_dict["groupby/router_depth"] = args.router_depth
            if hasattr(args, "router_dropout"):
                summary_log_dict["groupby/router_dropout"] = args.router_dropout
        wandb.log(summary_log_dict)

        # Create a summary table
        trial_data = []
        for i, (train_mae, val_mae, test_mae, energy) in enumerate(
            zip(train_accuracies, validation_accuracies, test_accuracies, energies)
        ):
            trial_data.append([i + 1, train_mae, val_mae, test_mae, energy])

        wandb.log(
            {
                "trials_table": wandb.Table(
                    columns=["Trial", "Train_MAE", "Val_MAE", "Test_MAE", "Energy"],
                    data=trial_data,
                )
            }
        )

        wandb.finish()

    # Log every time a dataset is completed
    df = pd.DataFrame(results)
    with open(
        f"results/graph_regression_{args.layer_type}_{args.encoding}.csv", "a"
    ) as f:
        df.to_csv(f, mode="a", header=f.tell() == 0)
