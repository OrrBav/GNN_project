# taken from: https://github.com/dmlc/dgl/blob/master/examples/pytorch/gin/train.py#L103
# uses torch conda env
import argparse

import numpy as np
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dgl.data import GINDataset
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import Dataset
import os
import pickle
import time


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
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        num_layers = 5
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
        self.drop = nn.Dropout(0.5)
        self.pool = (
            SumPooling()
        )  # change to mean readout (AvgPooling) on social network datasets

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


def split_fold10(labels, fold_idx=0):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, valid_idx = idx_list[fold_idx]
    return train_idx, valid_idx


def evaluate(dataloader, device, model):
    model.eval()
    total = 0
    total_correct = 0
    for batched_graph, labels in dataloader:
        batched_graph = batched_graph.to(device)
        labels = labels.to(device)
        feat = batched_graph.ndata.pop("attr")
        total += len(labels)
        logits = model(batched_graph, feat)
        _, predicted = torch.max(logits, 1)
        total_correct += (predicted == labels).sum().item()
    acc = 1.0 * total_correct / total
    return acc


def train(train_loader, val_loader, device, model):
    # loss function, optimizer and scheduler
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # training loop
    for epoch in range(350):
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
        train_acc = evaluate(train_loader, device, model)
        valid_acc = evaluate(val_loader, device, model)
        print(
            "Epoch {:05d} | Loss {:.4f} | Train Acc. {:.4f} | Validation Acc. {:.4f} ".format(
                epoch, total_loss / (batch + 1), train_acc, valid_acc
            )
        )


# Assuming CustomDGLDataset is already defined as previously discussed
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
            dgl_graph = dgl.from_networkx(G)
            dgl_graph = dgl.add_self_loop(dgl_graph)
            embeddings = [torch.tensor(G.nodes[n]['embedding'], dtype=torch.float32) for n in G.nodes()]
            dgl_graph.ndata['attr'] = torch.stack(embeddings).float()
            dgl_graph.y = torch.tensor(label, dtype=torch.long).squeeze()  # Ensure labels are 1D
            dgl_graph.name = filename
            dgl_graph_list.append(dgl_graph)
    return dgl_graph_list

def split_dataset(num_graphs, train_ratio=0.8):
    indices = list(range(num_graphs))
    split = int(train_ratio * num_graphs)
    train_idx, val_idx = indices[:split], indices[split:]
    return train_idx, val_idx


if __name__ == "__main__":
    start = time.time()
    print(f"Training with DGL built-in GINConv module with a fixed epsilon = 0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set directory to where the custom pkl files are stored
    dataset_dir = '/dsi/sbm/OrrBavly/colon_data/embedding_graphs_90th_perc'  
    print("Laoding custom graphs...")
    ## loades netx graphs from a dir. MAKE SURE to specify data type for correct lables.
    dgl_graph_list = load_custom_graphs(dataset_dir, data_type='colon')
    dataset = CustomDGLDataset(dgl_graph_list)

    labels = [label.item() for _, label in dataset]
    train_idx, val_idx = split_fold10(labels)

    print("Creating Dataloader...")
    # create dataloader
    train_loader = GraphDataLoader(
        dataset,
        sampler=SubsetRandomSampler(train_idx),
        batch_size=16,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = GraphDataLoader(
        dataset,
        sampler=SubsetRandomSampler(val_idx),
        batch_size=16,
        pin_memory=torch.cuda.is_available(),
    )

    # Create GIN model
    in_size = dataset[0][0].ndata['attr'].shape[1]  # Input feature size
    out_size = 2  # Assuming binary classification (0 or 1 for label)
    model = GIN(in_size, 16, out_size).to(device)

    # model training/validating
    print("Training...")
    train(train_loader, val_loader, device, model)
    
    end = time.time()
    print(f"Runtime: {(end - start)/60}")