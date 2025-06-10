# taken from: https://github.com/dmlc/dgl/blob/master/examples/pytorch/gin/train.py#L103
# uses torch conda env
import argparse

import numpy as np
import dgl
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dgl.data import GINDataset
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling
from dgl.nn import SAGEConv, SumPooling
from dgl.nn import GATConv

# from sklearn.model_selection import StratifiedKFold ### used manual function for split_fold10
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import Dataset
import os
import pickle
import time
import logging
from itertools import product


GPU_DEVICE = 1
DATASET_DIR = '/home/dsi/orrbavly/GNN_project/data/embedding_graphs_90th_perc_new' 
SUB_GRAPHS = False
K = 10
DATA_TYPE = 'ovarian'

HYPER_PARAMETER_TUNE = False
# Hyper parameters. is parentheses - original values.
EPOCHS = 350
NUM_LAYERS = 5 # (5)
LR = 0.01 # (0.01)
STEP_SIZE= 50 # (50)
GAMMA = 0.5 # (0.5)
HIDDEN_DIM = 16 # (16)
DROPOUT = 0.5 # (0.5)

def get_percentile_based_subgraphs(graph, lower_percentile=90, upper_percentile=99):
    """
    Extracts subgraphs based on component sizes within specified percentiles.
    """
    
    # Get all components and their sizes
    components = list(nx.connected_components(graph))
    component_sizes = np.array([len(c) for c in components])
    
    # Calculate size thresholds
    min_size = np.percentile(component_sizes, lower_percentile)
    max_size = np.percentile(component_sizes, upper_percentile)
    
    # Filter components by size range
    filtered_components = [
        component for component in components
        if min_size <= len(component) <= max_size
    ]
    
    # Combine selected subgraphs into a single graph
    combined_graph = nx.compose_all([graph.subgraph(component).copy() for component in filtered_components])
    return combined_graph

def get_top_k_components(graph, k=1):
    """
    Extracts the k largest connected components from a NetworkX graph.

    Parameters:
        graph (networkx.Graph): The input graph (assumes undirected).
        k (int): Number of largest components to extract.

    Returns:
        networkx.Graph: A graph containing only the k largest components.
    """
    # Get all connected components sorted by size (largest first)
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    
    # Get subgraphs for the top k components
    top_k_subgraphs = [graph.subgraph(component).copy() for component in components[:k]]
    
    # Combine the subgraphs into a single graph
    combined_graph = nx.compose_all(top_k_subgraphs)
    
    return combined_graph

class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))

    def forward(self, x):
        h = x
        h = F.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)


class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=DROPOUT, num_layers=NUM_LAYERS):
        super().__init__()
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        num_layers = num_layers
        self.drop = nn.Dropout(dropout)
        self.pool = (
            SumPooling()
        )  # change to mean readout (AvgPooling) on social network datasets

        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(num_layers - 1):  # excluding the input layer
            if layer == 0:
                mlp = MLP(input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(
                GINConv(mlp, learn_eps=False)
            )  # set to True if learning epsilon
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        # linear functions for graph sum poolings of output of each layer
        self.linear_prediction = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linear_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linear_prediction.append(nn.Linear(hidden_dim, output_dim))
 

    def forward(self, g, h):
        # list of hidden representation at each layer (including the input layer)
        hidden_rep = [h]
        for i, layer in enumerate(self.ginlayers):
            h = layer(g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
        score_over_layer = 0
        # perform graph sum pooling over all nodes in each layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linear_prediction[i](pooled_h))
        return score_over_layer


class GATClassifier(nn.Module):
    def __init__(self, in_feats, hidden_dim, out_dim, num_layers=3, num_heads=4, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.pool = SumPooling()

        # First layer
        self.layers.append(GATConv(in_feats, hidden_dim, num_heads))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim * num_heads))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(hidden_dim * num_heads, hidden_dim, num_heads))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim * num_heads))

        # Final GAT layer (no concatenation, single head for output)
        self.layers.append(GATConv(hidden_dim * num_heads, hidden_dim, num_heads=1))

        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, g, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(g, x)
            x = x.flatten(1)  # concatenate heads
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)

        # Final layer (no ReLU, no flatten because num_heads=1)
        x = self.layers[-1](g, x).squeeze(1)

        hg = self.pool(g, x)
        return self.classifier(hg)

class GraphSAGEClassifier(nn.Module):
    def __init__(self, in_feats, hidden_dim, out_dim, num_layers=3, dropout=0.5, aggregator_type='mean'):
        super().__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.pool = SumPooling()

        # Input layer
        self.layers.append(SAGEConv(in_feats, hidden_dim, aggregator_type))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_dim, hidden_dim, aggregator_type))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Output layer
        self.layers.append(SAGEConv(hidden_dim, hidden_dim, aggregator_type))

        # Final classifier after pooling
        self.linear = nn.Linear(hidden_dim, out_dim)

    def forward(self, g, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(g, x)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)

        # Final layer (no batch norm or ReLU)
        x = self.layers[-1](g, x)
        x = self.dropout(x)

        hg = self.pool(g, x)
        return self.linear(hg)

# def split_fold10(labels, fold_idx=0):
#     skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
#     idx_list = []
#     for idx in skf.split(np.zeros(len(labels)), labels):
#         idx_list.append(idx)
#     train_idx, valid_idx = idx_list[fold_idx]
#     return train_idx, valid_idx

def split_fold10_manual(labels, fold_idx=0, n_splits=10, random_seed=0):
    """
    Manual implementation for StratifiedKFold function from sklearn, because of conflicting issues when trying to install sklearn in conda env.
    """
    labels = np.array(labels)  # Ensure labels are a NumPy array
    # np.random.seed(random_seed) # Remove comment to ensure reproducibility
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    
    # Initialize fold indices
    fold_indices = [[] for _ in range(n_splits)]
    
    # Stratify by unique labels
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        np.random.shuffle(label_indices)
        
        # Split into folds
        fold_sizes = [len(label_indices) // n_splits] * n_splits
        for i in range(len(label_indices) % n_splits):
            fold_sizes[i] += 1
        
        current_idx = 0
        for i, size in enumerate(fold_sizes):
            fold_indices[i].extend(label_indices[current_idx:current_idx + size])
            current_idx += size
    
    # Combine folds for train/validation
    valid_idx = fold_indices[fold_idx]
    train_idx = np.concatenate([fold_indices[i] for i in range(n_splits) if i != fold_idx])
    
    return train_idx, valid_idx


def stratified_train_test_split(labels, test_size=0.2, seed=42):
    """
    Split indices into stratified train/test subsets. Focuses on train/test splits, not train/validation split (as split_fold10_manual function does).
    Returns: train_indices, test_indices
    """
    np.random.seed(seed)
    labels = np.array(labels)
    unique_labels = np.unique(labels)

    test_indices = []
    train_indices = []

    for label in unique_labels:
        idx = np.where(labels == label)[0]
        np.random.shuffle(idx)
        n_test = int(np.floor(test_size * len(idx)))
        test_indices.extend(idx[:n_test])
        train_indices.extend(idx[n_test:])

    return np.array(train_indices), np.array(test_indices)


def evaluate(dataloader, device, model):
    model.eval()
    total = 0
    total_correct = 0

    # For balanced accuracy
    class_correct = {}
    class_total = {}

    for batched_graph, labels in dataloader:
        batched_graph = batched_graph.to(device)
        labels = labels.to(device)
        feat = batched_graph.ndata.pop("attr")
        logits = model(batched_graph, feat)
        _, predicted = torch.max(logits, 1)

        total += len(labels)
        total_correct += (predicted == labels).sum().item()

        for label, pred in zip(labels, predicted):
            label = label.item()
            class_total[label] = class_total.get(label, 0) + 1
            if label == pred.item():
                class_correct[label] = class_correct.get(label, 0) + 1

    acc = total_correct / total

    recalls = []
    for cls in class_total:
        correct = class_correct.get(cls, 0)
        recall = correct / class_total[cls]
        recalls.append(recall)

    balanced_acc = sum(recalls) / len(recalls)

    # print(f"Overall Accuracy: {acc:.4f} | Balanced Accuracy: {balanced_acc:.4f}")
    return acc, balanced_acc


def train(train_loader, val_loader, device, model):
    # loss function, optimizer and scheduler
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    # training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch, (batched_graph, labels) in enumerate(train_loader):
            batched_graph = batched_graph.to(device)
            labels = labels.to(device)
            feat = batched_graph.ndata.pop("attr")
            logits = model(batched_graph, feat)
            loss = loss_fcn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        train_acc, train_balnc_acc = evaluate(train_loader, device, model)
        valid_acc, valid_balnc_acc = evaluate(val_loader, device, model)
        print(
            "Epoch {:05d} | Loss {:.4f} | Train Acc./balnc_Acc {:.4f}/{:.4f} | Validation Acc./balnc_Acc {:.4f}/{:.4f} ".format(
                epoch + 1, total_loss / (batch + 1), train_acc, train_balnc_acc, valid_acc, valid_balnc_acc
            )
        )

def train_kfold_val(dataset, train_val_idx, device, model, n_splits=3, learn_r=LR, epochs=EPOCHS):
    '''
    My addition, modified train function (based on original DGL function).
    '''
    labels = [dataset[i][1].item() for i in train_val_idx]

    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learn_r)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=0.5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        val_accuracies = []

        # Create k-folds for this epoch
        for fold_idx in range(n_splits):
            train_fold_idx, val_fold_idx = split_fold10_manual(labels, fold_idx, n_splits=n_splits)
            train_idx = train_val_idx[train_fold_idx]
            val_idx = train_val_idx[val_fold_idx]

            train_loader = GraphDataLoader(dataset, sampler=SubsetRandomSampler(train_idx), batch_size=16)
            val_loader = GraphDataLoader(dataset, sampler=SubsetRandomSampler(val_idx), batch_size=16)

            # Train on train fold
            for batched_graph, labels_ in train_loader:
                batched_graph = batched_graph.to(device)
                labels_ = labels_.to(device)
                feat = batched_graph.ndata.pop("attr")
                logits = model(batched_graph, feat)
                loss = loss_fcn(logits, labels_)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Validate on val fold
            val_acc, val_balnc_add = evaluate(val_loader, device, model)
            val_accuracies.append(val_balnc_add)

        scheduler.step()
        avg_val_acc = np.mean(val_accuracies)

        train_loader_full = GraphDataLoader(
            dataset,
            sampler=SubsetRandomSampler(train_val_idx),
            batch_size=16,
        )
        train_acc, train_balnc_acc = evaluate(train_loader_full, device, model)

        print(
            "Epoch {:05d} | Loss {:.4f} | Train Acc. {:.4f} | Train BalAcc. {:.4f} | Val BalAcc ({}-fold): {:.4f}".format(
                epoch + 1, total_loss / len(train_val_idx), train_acc, train_balnc_acc, n_splits, avg_val_acc,
            )
        )
    return avg_val_acc

class CustomDGLDataset(Dataset):
    def __init__(self, graph_list):
        self.graphs = graph_list

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        return graph, graph.y


# Function to load graphs from pkl files
def load_custom_graphs(directory, data_type='ovarian'):
    dgl_graph_list = []
    # controls the patients type used in experiment
    # ["fp", "nd", "nh"]
    invalid_files = ["fp"]
    for filename in os.listdir(directory):
        # maybe check files ends with 'pkl' if neccecary in future
        if all(invalid not in filename for invalid in invalid_files):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'rb') as f:
                G = pickle.load(f)  # Load NetworkX graph from pkl file
            if data_type == 'ovarian':
                label = 1 if 'OC' in filename else 0
            elif data_type == 'colon':
                label = 1 if 'high' in filename else 0
            if SUB_GRAPHS:
                G = get_top_k_components(G, k=K)
            dgl_graph = dgl.from_networkx(G)
            dgl_graph = dgl.add_self_loop(dgl_graph)
            embeddings = [torch.tensor(G.nodes[n]['embedding'], dtype=torch.float32) for n in G.nodes()]
            dgl_graph.ndata['attr'] = torch.stack(embeddings).float()
            dgl_graph.y = torch.tensor(label, dtype=torch.long).squeeze()  # Ensure labels are 1D
            dgl_graph.name = filename
            dgl_graph_list.append(dgl_graph)
    print(f"Number of graphs: {len(dgl_graph_list)}")
    return dgl_graph_list

def split_dataset(num_graphs, train_ratio=0.8):
    indices = list(range(num_graphs))
    split = int(train_ratio * num_graphs)
    train_idx, val_idx = indices[:split], indices[split:]
    return train_idx, val_idx


if __name__ == "__main__":
    start = time.time()
    ### MY addition
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cpu":
        logging.warning("CUDA is not available, using CPU for training.")
    else:
        # Print available devices
        num_devices = torch.cuda.device_count()
        logging.info(f"CUDA is available. Number of devices: {num_devices}")
        print(f"CUDA is available. Number of devices: {num_devices}")

        # Try connecting to the specific device
        try:
            torch.cuda.set_device(GPU_DEVICE)  # SET GPU INDEX HERE:
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            logging.info(f"Using GPU device {current_device}: {device_name}")
            print(f"Using GPU device {current_device}: {device_name}")
        except Exception as e:
            logging.error(f"Failed to connect to GPU: {e}")
            device = torch.device("cpu")

    print(f"Training with DGL built-in GINConv module with a fixed epsilon = 0")
    print(f"Using device: {device}")

    # print(f"Training with DGL built-in GINConv module with a fixed epsilon = 0")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    
    print("Laoding custom graphs...")
    print(f"Loading Components: {SUB_GRAPHS}")
    ## loades netx graphs from a dir. MAKE SURE to specify data type for correct lables.
    dgl_graph_list = load_custom_graphs(DATASET_DIR, data_type=DATA_TYPE)
    dataset = CustomDGLDataset(dgl_graph_list)

    labels = [label.item() for _, label in dataset]
    # Step 1: Stratified fixed split into train+val and test
    train_val_idx, test_idx = stratified_train_test_split(labels, test_size=0.2)

    # Step 2: K-fold CV on train+val
    train_val_labels = [labels[i] for i in train_val_idx]
    cv_train_idx, cv_val_idx = split_fold10_manual(train_val_labels, fold_idx=0)

    # Step 3: Map indices back to full dataset
    train_idx = train_val_idx[cv_train_idx]
    val_idx = train_val_idx[cv_val_idx]

    print("Creating Dataloader...")
    # create dataloader
    # ***NOTE*** when using costume kfold train function, train_loader and val_loader are not needed - as they are defined inside train fuctnion.

    # train_loader = GraphDataLoader(
    #     dataset,
    #     sampler=SubsetRandomSampler(train_idx),
    #     batch_size=16,
    #     pin_memory=torch.cuda.is_available(),
    # )
    # val_loader = GraphDataLoader(
    #     dataset,
    #     sampler=SubsetRandomSampler(val_idx),
    #     batch_size=16,
    #     pin_memory=torch.cuda.is_available(),
    # )

    test_loader = GraphDataLoader(
    dataset,
    sampler=SubsetRandomSampler(test_idx),
    batch_size=16,
    pin_memory=torch.cuda.is_available(),
    )

    # Create GIN model
    in_size = dataset[0][0].ndata['attr'].shape[1]  # Input feature size
    out_size = 2  # binary classification

    if HYPER_PARAMETER_TUNE:
        print("Running Hyper Parameters Tuning:")
        # ~~~~~~~ HYPER PARAMETERS TUNE ~~~~~~~~
        best_acc = 0
        best_config = None

        hidden_dims = [16, 32]
        lrs = [0.01, 0.005]
        dropouts = [0.3, 0.5]
        num_layers = [3, 5, 7]
        # epohcs = 30
        param_grid = list(product(hidden_dims, lrs, dropouts, num_layers))
        for hidden_dim, lr, dropout, num_layer in param_grid:
            print(f"\nTesting config: hidden={hidden_dim}, lr={lr}, dropout={dropout}, num_layers={num_layer}")
            model = GIN(in_size, hidden_dim, out_size, dropout=dropout, num_layers=num_layer).to(device)
            avg_val_acc = train_kfold_val(dataset, train_val_idx, device, model, n_splits=3, learn_r=lr, epochs=30)

            if avg_val_acc > best_acc:
                best_acc = avg_val_acc
                best_config = (hidden_dim, lr, dropout, num_layer)

        print(f"\nBest config: {best_config} with acc {best_acc:.4f}")

        ### RUN new gin model on best hyper parameters.
        print("Training GIN Model...")
        best_hidden, best_lr, best_dropout, best_layers = best_config
        model = GIN(in_size, best_hidden, out_size, dropout=best_dropout, num_layers=best_layers).to(device)
        
        # train(train_loader, val_loader, device, model)
        train_kfold_val(dataset, train_val_idx, device, model, n_splits=3, learn_r=best_lr)
    
    else:
        # Run Gin model on pre determined hyper parameters:
        print("Training Graphsage Model...")
        # model = GIN(in_size, HIDDEN_DIM, out_size).to(device)
        # model = GraphSAGEClassifier(in_feats=in_size, hidden_dim=64, out_dim=out_size,
        #                     num_layers=5, dropout=0.3, aggregator_type='mean').to(device)
        model = GATClassifier(in_feats=in_size, hidden_dim=64, out_dim=out_size,
                            num_layers=5, num_heads=4, dropout=0.5).to(device)
 
        # train(train_loader, val_loader, device, model)
        train_kfold_val(dataset, train_val_idx, device, model, n_splits=3, learn_r=0.01)

    # Evaluate the model
    test_acc, test_baln_acc = evaluate(test_loader, device, model)
    print(f"Test Accuracy: {test_acc:.4f}\nTest Balanced Accuracy: {test_baln_acc:.4f}") 
    end = time.time()
    print(f"Script Runtime: {(end - start)/60}")
    if HYPER_PARAMETER_TUNE:
        print(f"\nBest config: (hidden layers size, lr, dropput, num_layers) {best_config} with train/val acc {best_acc:.4f}")