"""Script to run graph classification experiments."

This script orchestrates comprehensive graph classification experiments across diverse datasets
including molecular graphs (MUTAG, ENZYMES, PROTEINS), social networks (IMDB, COLLAB, REDDIT),
and computer vision tasks (MNIST, CIFAR10 superpixels). It supports various GNN architectures.

The script handles dataset preprocessing, optional structural encodings (Laplacian eigenvectors,
random walk features, curvature profiles), multi-trial training with statistical analysis,
and comprehensive result logging for benchmarking different graph neural network approaches.
"""

from attrdict import AttrDict
from torch_geometric.datasets import TUDataset, MoleculeNet
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx, to_dense_adj
import torch_geometric.transforms as T

from torch_geometric.datasets import GNNBenchmarkDataset

# import custom encodings
from torchvision.transforms import Compose
from custom_encodings import LocalCurvatureProfile, AltLocalCurvatureProfile

from experiments.graph_classification import Experiment

from tqdm import tqdm

import time
import torch
import numpy as np
import pandas as pd
from hyperparams import get_args_from_input

import pickle
import wget
import zipfile
import os

from torch_geometric.datasets import GNNBenchmarkDataset

import wandb
import os
from pathlib import Path


def _convert_lrgb(dataset: torch.Tensor) -> torch.Tensor:
    x = dataset[0]
    edge_attr = dataset[1]
    edge_index = dataset[2]
    y = dataset[3]

    return Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)


# Auto-detect if we're on cluster or local
if os.path.exists("/n/netscratch/mweber_lab/Lab/graph_datasets"):
    data_directory = "/n/netscratch/mweber_lab/Lab/graph_datasets"
    print("📁 Using cluster data directory")
else:
    data_directory = "./graph_datasets"
    print("📁 Using local data directory")
    os.makedirs(data_directory, exist_ok=True)

# Create data subdirectory for encodings
os.makedirs("data", exist_ok=True)
print(f"📁 Raw datasets: {data_directory}")
print(f"💾 Encoded datasets: ./data/")

# New datasets
print("📊 Loading NEW benchmark datasets...")
print("  ⏳ Loading MNIST superpixel graphs...")
mnist = list(GNNBenchmarkDataset(root=data_directory, name="MNIST"))
print(f"  ✅ MNIST loaded: {len(mnist)} graphs")

print("  ⏳ Loading CIFAR10 superpixel graphs...")
cifar = list(GNNBenchmarkDataset(root=data_directory, name="CIFAR10"))
print(f"  ✅ CIFAR10 loaded: {len(cifar)} graphs")

print("  ⏳ Loading PATTERN synthetic graphs...")
pattern = list(GNNBenchmarkDataset(root=data_directory, name="PATTERN"))
print(f"  ✅ PATTERN loaded: {len(pattern)} graphs")

print("📊 Loading existing TU datasets...")
# import TU datasets
print("  ⏳ Loading MUTAG...")
mutag = list(TUDataset(root=data_directory, name="MUTAG"))
print(f"  ✅ MUTAG loaded: {len(mutag)} graphs")

print("  ⏳ Loading ENZYMES...")
enzymes = list(TUDataset(root=data_directory, name="ENZYMES"))
print(f"  ✅ ENZYMES loaded: {len(enzymes)} graphs")

print("  ⏳ Loading PROTEINS...")
proteins = list(TUDataset(root=data_directory, name="PROTEINS"))
print(f"  ✅ PROTEINS loaded: {len(proteins)} graphs")

print("  ⏳ Loading IMDB-BINARY...")
imdb = list(TUDataset(root=data_directory, name="IMDB-BINARY"))
print(f"  ✅ IMDB-BINARY loaded: {len(imdb)} graphs")

print("  ⏳ Loading COLLAB...")
collab = list(TUDataset(root=data_directory, name="COLLAB"))
print(f"  ✅ COLLAB loaded: {len(collab)} graphs")

print("  ⏳ Loading REDDIT-BINARY...")
reddit = list(TUDataset(root=data_directory, name="REDDIT-BINARY"))
print(f"  ✅ REDDIT-BINARY loaded: {len(reddit)} graphs")

print("🎉 All datasets loaded successfully!")

"""
# import peptides-func dataset
peptides_zip_filepath = data_directory
peptides_train = torch.load(os.path.join(peptides_zip_filepath, "peptidesfunc", "train.pt"))
peptides_val = torch.load(os.path.join(peptides_zip_filepath, "peptidesfunc", "val.pt"))
peptides_test = torch.load(os.path.join(peptides_zip_filepath, "peptidesfunc", "test.pt"))
peptides_func = [_convert_lrgb(peptides_train[i]) for i in range(len(peptides_train))] + [_convert_lrgb(peptides_val[i]) for i in range(len(peptides_val))] + [_convert_lrgb(peptides_test[i]) for i in range(len(peptides_test))]
"""


datasets = {
    "mutag": mutag,
    "enzymes": enzymes,
    "proteins": proteins,
    "imdb": imdb,
    "collab": collab,
    "reddit": reddit,
    # New datasets:
    "mnist": mnist,
    "cifar": cifar,
    "pattern": pattern,
}
# datasets = {"collab": collab, "imdb": imdb, "proteins": proteins, "reddit": reddit}


for key in datasets:
    if key in ["reddit", "imdb", "collab"]:
        for graph in datasets[key]:
            n = graph.num_nodes
            graph.x = torch.ones((n, 1))


def log_to_file(message, filename="results/graph_classification.txt"):
    print(message)
    file = open(filename, "a")
    file.write(message)
    file.close()


default_args = AttrDict(
    {
        "dropout": 0.1,
        "num_layers": 4,
        "hidden_dim": 64,
        "learning_rate": 1e-3,
        "layer_type": "MoE",
        "display": True,
        "num_trials": 10,
        "eval_every": 1,
        "patience": 50,
        "output_dim": 2,
        "alpha": 0.1,
        "eps": 0.001,
        "dataset": None,
        "last_layer_fa": False,
        "encoding": None,
        "mlp": True,
        "layer_types": None,
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
    "mutag": AttrDict({"output_dim": 2}),
    "enzymes": AttrDict({"output_dim": 6}),
    "proteins": AttrDict({"output_dim": 2}),
    "collab": AttrDict({"output_dim": 3}),
    "imdb": AttrDict({"output_dim": 2}),
    "reddit": AttrDict({"output_dim": 2}),
    "peptides": AttrDict({"output_dim": 10}),
    # New datasets:
    "mnist": AttrDict({"output_dim": 10}),
    "cifar": AttrDict({"output_dim": 10}),
    "pattern": AttrDict({"output_dim": 2}),  # Binary classification
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
            print(f"✅ ENCODING ALREADY EXISTS: Loading {key}_{args.encoding}.pt")
            dataset = torch.load(f"data/{key}_{args.encoding}.pt")

        elif args.encoding == "LCP":
            print(f"🔄 ENCODING STARTED: {args.encoding} for {key.upper()}...")
            lcp = LocalCurvatureProfile()
            for i in tqdm(
                range(len(dataset)), desc=f"Encoding {key.upper()} with {args.encoding}"
            ):
                dataset[i] = lcp.compute_orc(dataset[i])
                print(f"Graph {i} of {len(dataset)} encoded with {args.encoding}")
            torch.save(dataset, f"data/{key}_{args.encoding}.pt")
            print(f"💾 Encoded dataset saved to: data/{key}_{args.encoding}.pt")

        else:
            print("ENCODING STARTED...")
            org_dataset_len = len(dataset)
            drop_datasets = []
            current_graph = 0

            for i in tqdm(
                range(org_dataset_len),
                desc=f"Encoding {key.upper()} with {args.encoding}",
            ):
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
                    tqdm.write(
                        f"⚠️  Graph {current_graph} dropped due to encoding error"
                    )
                    drop_datasets.append(i)
                    current_graph += 1

            for i in sorted(drop_datasets, reverse=True):
                dataset.pop(i)

            # save the dataset to a file in the data folder
            torch.save(dataset, f"data/{key}_{args.encoding}.pt")
            print(f"💾 Encoded dataset saved to: data/{key}_{args.encoding}.pt")

    # create a dictionary of the graphs in the dataset with the key being the graph index
    graph_dict = {}
    for i in range(len(dataset)):
        graph_dict[i] = []
    print("GRAPH DICTIONARY CREATED...")

    print(f"🚀 TRAINING STARTED for {key.upper()} dataset...")
    print(
        f"🔧 Model: {args.layer_type} | Layers: {args.num_layers} | Hidden: {args.hidden_dim}"
    )
    start = time.time()

    # Add progress bar for trials
    if args.wandb_enabled:
        # Initialize wandb once for the entire run (all trials)
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=dict(args),
            dir=args.wandb_dir,
            tags=args.wandb_tags,
        )
        print(f"🚀 WandB initialized: {wandb.run.name}")

    for trial in tqdm(
        range(args.num_trials), desc=f"Training {key.upper()}", unit="trial"
    ):
        print(f"\n📊 TRIAL {trial + 1}/{args.num_trials}")
        train_acc, validation_acc, test_acc, energy, dictionary = Experiment(
            args=args, dataset=dataset
        ).run()
        train_accuracies.append(train_acc)
        validation_accuracies.append(validation_acc)
        test_accuracies.append(test_acc)
        energies.append(energy)

        # Show intermediate results
        print(
            f"   Train: {train_acc:.3f} | Val: {validation_acc:.3f} | Test: {test_acc:.3f}"
        )

        # Log trial results to wandb
        if args.wandb_enabled:
            wandb.log(
                {
                    "trial/train_acc": train_acc,
                    "trial/val_acc": validation_acc,
                    "trial/test_acc": test_acc,
                    "trial/energy": energy,
                    "trial_num": trial + 1,
                }
            )

        for name in dictionary.keys():
            if dictionary[name] != -1:
                graph_dict[name].append(dictionary[name])

    end = time.time()
    run_duration = end - start
    print(f"⏱️  Training completed in {run_duration:.2f} seconds")

    # with open(f"results/{args.num_layers}_layers/{key}_{args.layer_type}_{args.encoding}_graph_dict.pickle", "wb") as f:
    # pickle.dump(graph_dict, f)
    # print(f"Graph dictionary for {key} pickled")

    train_mean = 100 * np.mean(train_accuracies)
    val_mean = 100 * np.mean(validation_accuracies)
    test_mean = 100 * np.mean(test_accuracies)
    energy_mean = 100 * np.mean(energies)
    train_ci = 2 * np.std(train_accuracies) / (args.num_trials**0.5)
    val_ci = 2 * np.std(validation_accuracies) / (args.num_trials**0.5)
    test_ci = 2 * np.std(test_accuracies) / (args.num_trials**0.5)
    energy_ci = 200 * np.std(energies) / (args.num_trials**0.5)
    log_to_file(f"RESULTS FOR {key} ({args.layer_type}), with {args.encoding}:\n")
    log_to_file(f"average acc: {test_mean}\n")
    log_to_file(f"plus/minus:  {test_ci}\n\n")
    results.append(
        {
            "dataset": key,
            "encoding": args.encoding,
            "layer_type": args.layer_type,
            "layer_types": args.layer_types,
            "encoding": args.encoding,
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

    # Log final summary to wandb
    if args.wandb_enabled:
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
            }
        )
        wandb.finish()

    # Log every time a dataset is completed
    df = pd.DataFrame(results)
    with open(
        f"results/graph_classification_{args.layer_type}_{args.encoding}.csv", "a"
    ) as f:
        df.to_csv(f, mode="a", header=f.tell() == 0)

    print(f"\n🎯 FINAL RESULTS for {key.upper()}:")
    print(f"   📈 Test Accuracy: {test_mean:.2f}% ± {test_ci:.2f}%")
    print(f"   📊 Validation Accuracy: {val_mean:.2f}% ± {val_ci:.2f}%")
    print(f"   🏃 Training Accuracy: {train_mean:.2f}% ± {train_ci:.2f}%")
    print(f"   ⚡ Energy: {energy_mean:.2f}% ± {energy_ci:.2f}%")
    print(f"   ⏱️  Duration: {run_duration:.2f}s")
    print(
        f"   💾 Results saved to: results/graph_classification_{args.layer_type}_{args.encoding}.csv"
    )
