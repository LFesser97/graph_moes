import torch
import os
import copy
import numpy as np
from attrdict import AttrDict
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from math import inf

import random
from torch.utils.data import Dataset, Subset

from models.graph_model import GNN

default_args = AttrDict(
    {"learning_rate": 1e-3,
    "max_epochs": 1000000,
    "display": True,
    "device": None,
    "eval_every": 1,
    "stopping_criterion": "train",
    "stopping_threshold": 1.01,
    "patience": 30,
    "train_fraction": 0.8,
    "validation_fraction": 0.1,
    "test_fraction": 0.1,
    "dropout": 0.0,
    "weight_decay": 1e-5,
    "input_dim": None,
    "hidden_dim": 32,
    "output_dim": 1,
    "hidden_layers": None,
    "num_layers": 1,
    "batch_size": 10,
    "layer_type": "R-GCN",
    "num_relations": 2,
    "last_layer_fa": False
    }
    )

class Experiment:
    def __init__(self, args=None, dataset=None, train_dataset=None, validation_dataset=None, test_dataset=None):
        self.args = default_args + args
        self.dataset = dataset
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.categories = None
        self.datasplits = None

        if self.args.device is None:
            self.args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.args.hidden_layers is None:
            self.args.hidden_layers = [self.args.hidden_dim] * self.args.num_layers
        if self.args.input_dim is None:
            self.args.input_dim = self.dataset[0].x.shape[1]
        for graph in self.dataset:
            if not "edge_type" in graph.keys:
                num_edges = graph.edge_index.shape[1]
                graph.edge_type = torch.zeros(num_edges, dtype=int)
        if self.args.num_relations is None:
            if self.args.rewiring == "None":
                self.args.num_relations = 1
            else:
                self.args.num_relations = 2


        self.model = GNN(self.args).to(self.args.device)
       
        if self.test_dataset is None:
            dataset_size = min(1000, len(self.dataset))
            train_size = int(0.125 * dataset_size)

            # print the feature dimension of the first graph in the dataset
            print("Feature dimension of the first graph in the dataset: ", self.dataset[0].x.shape[1])
            
            # print the number of edges of the first graph in the dataset
            print("Number of edges of the first graph in the dataset: ", self.dataset[0].edge_index.shape[1])

            # sort all graphs in the dataset by number of nodes
            sorted_dataset = sorted(self.dataset, key=lambda x: x.num_nodes)
            
            # split the dataset into 8 parts, with the smallest graphs in the first part
            self.datasplits = {i: sorted_dataset[i*train_size:(i+1)*train_size] for i in range(8)}
            self.train_dataset = self.datasplits[7]
            self.validation_dataset = self.datasplits[1]
            self.test_dataset = self.datasplits[2]

        
    def run(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer)

        best_validation_acc = 0.0
        best_train_acc = 0.0
        best_test_acc = 0.0
        train_goal = 0.0
        epochs_no_improve = 0

        train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
        validation_loader = DataLoader(self.validation_dataset, batch_size=self.args.batch_size, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=self.args.batch_size, shuffle=True)
        # complete_loader = DataLoader(self.dataset, batch_size=self.args.batch_size, shuffle=True)
        complete_loader = DataLoader(self.dataset, batch_size=1)

        # create a dictionary of the graphs in the dataset with the key being the graph index
        graph_dict = {i: -1 for i in range(8)}

        for epoch in range(1, 1 + self.args.max_epochs):
            self.model.train()
            total_loss = 0
            optimizer.zero_grad()

            for graph in train_loader:
                graph = graph.to(self.args.device)
                y = graph.y.to(self.args.device)

                out = self.model(graph)
                loss = self.loss_fn(input=out, target=y)
                total_loss += loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            new_best_str = ''
            scheduler.step(total_loss)
            if epoch % self.args.eval_every == 0:
                train_acc = self.eval(loader=train_loader)

                if self.args.stopping_criterion == "train":
                    if train_acc > train_goal:
                        best_train_acc = train_acc
                        epochs_no_improve = 0
                        train_goal = train_acc * self.args.stopping_threshold
                        new_best_str = ' (new best train)'
                    elif train_acc > best_train_acc:
                        best_train_acc = train_acc
                        epochs_no_improve += 1
                    else:
                        epochs_no_improve += 1
                if self.args.display:
                    print(f'Epoch {epoch}, Train acc: {train_acc}')
                if epochs_no_improve > self.args.patience:
                    if self.args.display:
                        print(f'{self.args.patience} epochs without improvement, stopping training')
                        print(f'Best train acc: {best_train_acc}')
                        energy = 0

                        # evaluate the model on all datasets in datasplits
                        for i in range(8):
                            self.test_dataset = self.datasplits[i]
                            test_loader = DataLoader(self.test_dataset, batch_size=self.args.batch_size, shuffle=True)
                            test_acc = self.eval(loader=test_loader)
                            graph_dict[i] = test_acc
                            print(f'Test acc for dataset {i}: {test_acc}')

                    return best_train_acc, best_validation_acc, best_test_acc, energy, graph_dict
                
        if self.args.display:
            print('Reached max epoch count, stopping training')
            print(f'Best train acc: {best_train_acc}')

        energy = 0
        return best_train_acc, best_validation_acc, best_test_acc, energy, graph_dict

    def eval(self, loader):
        self.model.eval()
        sample_size = len(loader.dataset)
        with torch.no_grad():
            total_correct = 0
            for graph in loader:
                graph = graph.to(self.args.device)
                y = graph.y.to(self.args.device)
                out = self.model(graph)
                _, pred = out.max(dim=1)
                total_correct += pred.eq(y).sum().item()
                
        return total_correct / sample_size
    
    def check_dirichlet(self, loader):
        self.model.eval()
        sample_size = len(loader.dataset)
        with torch.no_grad():
            total_energy = 0
            for graph in loader:
                graph = graph.to(self.args.device)
                total_energy += self.model(graph, measure_dirichlet=True)
        return total_energy / sample_size


def custom_random_split(dataset, percentages):
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
    categories = [shuffled_list[:lengths[0]],
                  shuffled_list[lengths[0]:lengths[0]+lengths[1]],
                  shuffled_list[lengths[0]+lengths[1]:]]
    
    train_dataset = Subset(dataset, categories[0])
    validation_dataset = Subset(dataset, categories[1])
    test_dataset = Subset(dataset, categories[2])
    
    return train_dataset, validation_dataset, test_dataset, categories
