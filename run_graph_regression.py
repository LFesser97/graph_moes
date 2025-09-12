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
from attrdict import AttrDict
from torch_geometric.data import Data
from torch_geometric.datasets import ZINC

# import custom encodings

# Add wandb imports
import wandb
from custom_encodings import LocalCurvatureProfile
from experiments.graph_regression import Experiment
from hyperparams import get_args_from_input


def _convert_lrgb(dataset: torch.Tensor) -> torch.Tensor:
    x = dataset[0]
    edge_attr = dataset[1]
    edge_index = dataset[2]
    y = dataset[3]

    return Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)


# load zinc
data_directory = "/n/netscratch/mweber_lab/Lab/graph_datasets"
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

# datasets = {"zinc": zinc, "peptides_struct": peptides_struct}
datasets = {"zinc": zinc}


def log_to_file(message, filename="results/graph_regression.txt"):
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
        "num_trials": 15,
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
        "wandb_project": "MOE",
        "wandb_entity": "weber-geoml-harvard-university",
        "wandb_name": None,
        "wandb_dir": "./wandb",
        "wandb_tags": None,
    }
)

hyperparams = {
    "zinc": AttrDict({"output_dim": 1}),
    "peptides_struct": AttrDict({"output_dim": 11}),
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

                except:
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
    graph_dict = {}
    for i in range(len(dataset)):
        graph_dict[i] = []
    print("GRAPH DICTIONARY CREATED...")

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

    for trial in range(args.num_trials):
        print(f"Trial {trial + 1} of {args.num_trials}")

        # Initialize wandb for this specific trial
        if args.wandb_enabled:
            trial_run_name = f"trial_{trial + 1:02d}"
            if args.wandb_name:
                trial_run_name = f"{args.wandb_name}_trial_{trial + 1:02d}"

            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=trial_run_name,
                group=experiment_id,  # Group all trials together
                config={
                    **dict(args),
                    "trial_num": trial + 1,
                    "dataset": key,
                },
                dir=args.wandb_dir,
                tags=list(args.wandb_tags or []) + [f"trial_{trial + 1}"],
                reinit=True,  # Allow multiple runs in same process
            )
            print(f"🚀 WandB Trial Run: {wandb.run.name}")

        try:
            train_acc, validation_acc, test_acc, energy, dictionary = Experiment(
                args=args, dataset=dataset
            ).run()

            train_accuracies.append(train_acc)
            validation_accuracies.append(validation_acc)
            test_accuracies.append(test_acc)
            energies.append(energy)

            # Log final trial results to this trial's wandb run
            if args.wandb_enabled:
                wandb.log(
                    {
                        "final/train_mae": train_acc,
                        "final/val_mae": validation_acc,
                        "final/test_mae": test_acc,
                        "final/energy": energy,
                    }
                )

            for name in dictionary.keys():
                if dictionary[name] != -1:
                    graph_dict[name].append(dictionary[name])

        finally:
            # Finish this trial's wandb run
            if args.wandb_enabled:
                wandb.finish()

    end = time.time()
    run_duration = end - start

    with open(
        f"results/{args.num_layers}_layers/{key}_{args.layer_type}_{args.encoding}_graph_dict.pickle",
        "wb",
    ) as f:
        pickle.dump(graph_dict, f)
    print(f"Graph dictionary for {key} pickled")

    train_mean = np.mean(train_accuracies)
    val_mean = np.mean(validation_accuracies)
    test_mean = np.mean(test_accuracies)
    energy_mean = 100 * np.mean(energies)
    train_ci = 2 * np.std(train_accuracies) / (args.num_trials**0.5)
    val_ci = 2 * np.std(validation_accuracies) / (args.num_trials**0.5)
    test_ci = 2 * np.std(test_accuracies) / (args.num_trials**0.5)
    energy_ci = 200 * np.std(energies) / (args.num_trials**0.5)
    log_to_file(f"RESULTS FOR {key} ({args.layer_type}), with {args.encoding}:\n")
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
        }
    )

    # Create a summary run for the overall experiment
    if args.wandb_enabled:
        summary_run_name = "SUMMARY"
        if args.wandb_name:
            summary_run_name = f"{args.wandb_name}_SUMMARY"

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=summary_run_name,
            group=experiment_id,
            config={
                **dict(args),
                "dataset": key,
                "run_type": "summary",
                "num_trials": args.num_trials,
            },
            dir=args.wandb_dir,
            tags=list(args.wandb_tags or []) + ["summary"],
            reinit=True,
        )

        # Log aggregate statistics
        wandb.log(
            {
                "summary/test_mean": test_mean,
                "summary/test_ci": test_ci,
                "summary/val_mean": val_mean,
                "summary/val_ci": val_ci,
                "summary/train_mean": train_mean,
                "summary/train_ci": train_ci,
                "summary/energy_mean": energy_mean,
                "summary/energy_ci": energy_ci,
                "summary/run_duration": run_duration,
                # Log individual trial results for analysis
                "trials/train_maes": train_accuracies,
                "trials/val_maes": validation_accuracies,
                "trials/test_maes": test_accuracies,
                "trials/energies": energies,
            }
        )

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
