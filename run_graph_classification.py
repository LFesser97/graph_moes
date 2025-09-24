"""Script to run graph classification experiments."

This script orchestrates comprehensive graph classification experiments across diverse datasets
including molecular graphs (MUTAG, ENZYMES, PROTEINS), social networks (IMDB, COLLAB, REDDIT),
and computer vision tasks (MNIST, CIFAR10 superpixels). It supports various GNN architectures.

The script handles dataset preprocessing, optional structural encodings (Laplacian eigenvectors,
random walk features, curvature profiles), multi-trial training with statistical analysis,
and comprehensive result logging for benchmarking different graph neural network approaches.
"""

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from attrdict import AttrDict
from torch_geometric.data import Data

# from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.datasets import GNNBenchmarkDataset, TUDataset
from torch_geometric.datasets.lrgb import LRGBDataset

# import custom encodings
from tqdm import tqdm

import wandb
from custom_encodings import LocalCurvatureProfile
from experiments.graph_classification import Experiment
from hyperparams import get_args_from_input


def _convert_lrgb(dataset: torch.Tensor) -> torch.Tensor:
    x = dataset[0]
    edge_attr = dataset[1]
    edge_index = dataset[2]
    y = dataset[3]

    return Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)


# Simple approach - always use local directory
data_directory = (
    "/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/graph_moes/graph_datasets"
)
print("📁 Using project data directory")
os.makedirs(data_directory, exist_ok=True)

# Create data subdirectory for encodings
os.makedirs("data", exist_ok=True)
print(f"📁 Raw datasets: {data_directory}")
print("💾 Encoded datasets: ./data/")

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

print("and yet more...")

# Add to run_graph_classification.py
# print("  ⏳ Loading ogbg-molhiv...")
# molhiv = PygGraphPropPredDataset(name="ogbg-molhiv", root=data_directory)
# print(f"  ✅ ogbg-molhiv loaded: {len(molhiv)} graphs")

# print("  ⏳ Loading ogbg-molpcba...")
# molpcba = PygGraphPropPredDataset(name="ogbg-molpcba", root=data_directory)
# print(f"  ✅ ogbg-molpcba loaded: {len(molpcba)} graphs")

print("  ⏳ Loading Cluster...")
cluster = LRGBDataset(root=data_directory, name="Cluster")
print(f"  ✅ Cluster loaded: {len(cluster)} graphs")

print("  ⏳ Loading PascalVOC-SP...")
pascalvoc = LRGBDataset(root=data_directory, name="pascalvoc-sp")
print(f"  ✅ PascalVOC-SP loaded: {len(pascalvoc)} graphs")

print("  ⏳ Loading COCO-SP...")
coco = LRGBDataset(root=data_directory, name="coco-sp")
print(f"  ✅ COCO-SP loaded: {len(coco)} graphs")

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
    # LRGB datasets:
    "cluster": cluster,
    "pascalvoc": pascalvoc,
    "coco": coco,
    # OGB datasets:
    "molhiv": molhiv,
    "molpcba": molpcba,
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
    # LRGB datasets:
    "cluster": AttrDict({"output_dim": 6}),  # 6 clusters
    "pascalvoc": AttrDict({"output_dim": 21}),  # 21 object classes
    "coco": AttrDict({"output_dim": 81}),  # 81 object classes
    # OGB datasets:
    "molhiv": AttrDict(
        {"output_dim": 2}
    ),  # Binary classification (HIV active/inactive)
    "molpcba": AttrDict({"output_dim": 128}),  # Multi-label classification (128 assays)
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
    experiment_id = None
    if args.wandb_enabled:
        # Create a unique experiment group ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{key}_{args.layer_type}_{timestamp}"
        if args.wandb_name:
            experiment_id = f"{args.wandb_name}_{timestamp}"
        print(f"🔬 WandB Experiment Group: {experiment_id}")

    for trial in tqdm(
        range(args.num_trials), desc=f"Training {key.upper()}", unit="trial"
    ):
        print(f"\n📊 TRIAL {trial + 1}/{args.num_trials}")

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
                    # Add grouping variables:
                    "dataset_name": key,  # For grouping by dataset
                    "is_moe": args.layer_types
                    is not None,  # Boolean for MoE vs single layer
                    "model_type": (
                        "MoE" if args.layer_types is not None else args.layer_type
                    ),
                    "num_layers": args.num_layers,
                    "layer_combination": (
                        str(args.layer_types) if args.layer_types else args.layer_type
                    ),
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

            # Show intermediate results
            print(
                f"   Train: {train_acc:.3f} | Val: {validation_acc:.3f} | Test: {test_acc:.3f}"
            )

            # Log final trial results to this trial's wandb run
            if args.wandb_enabled:
                wandb.log(
                    {
                        "final/train_acc": train_acc,
                        "final/val_acc": validation_acc,
                        "final/test_acc": test_acc,
                        "final/energy": energy,
                        # Add grouping variables:
                        "groupby/dataset": key,
                        "groupby/model_type": (
                            "MoE" if args.layer_types else args.layer_type
                        ),
                        "groupby/is_moe": args.layer_types is not None,
                        "groupby/num_layers": args.num_layers,
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
                "trials/train_accs": train_accuracies,
                "trials/val_accs": validation_accuracies,
                "trials/test_accs": test_accuracies,
                "trials/energies": energies,
                # Add grouping variables:
                "groupby/dataset": key,
                "groupby/model_type": "MoE" if args.layer_types else args.layer_type,
                "groupby/is_moe": args.layer_types is not None,
                "groupby/num_layers": args.num_layers,
            }
        )

        # Create a summary table
        trial_data = []
        for i, (train_acc, val_acc, test_acc, energy) in enumerate(
            zip(train_accuracies, validation_accuracies, test_accuracies, energies)
        ):
            trial_data.append([i + 1, train_acc, val_acc, test_acc, energy])

        wandb.log(
            {
                "trials_table": wandb.Table(
                    columns=["Trial", "Train_Acc", "Val_Acc", "Test_Acc", "Energy"],
                    data=trial_data,
                )
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
