import argparse
import json
import logging
import os
from time import time

import dgl

import torch
import torch.nn
import torch.nn.functional as F
from dgl.data import LegacyTUDataset
from dgl.dataloading import GraphDataLoader
from networks import HGPSLModel
from torch.utils.data import random_split
from utils import get_stats

from torch.utils.data import Dataset
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description="HGP-SL-DGL")
    parser.add_argument(
        "--dataset",
        type=str,
        default="DD",
        choices=["DD", "PROTEINS", "NCI1", "NCI109", "Mutagenicity", "ENZYMES"],
        help="DD/PROTEINS/NCI1/NCI109/Mutagenicity/ENZYMES",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="batch size"  ## changed from 512 to 32 to 4
    )
    parser.add_argument(
        "--sample", type=str, default="true", help="use sample method"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-3, help="weight decay"
    )
    parser.add_argument(
        "--pool_ratio", type=float, default=0.3, help="pooling ratio" ## changed from 0.5 to 0.3
    )
    parser.add_argument("--hid_dim", type=int, default=64, help="hidden size")  ## changed from 128 to 64
    parser.add_argument(
        "--conv_layers", type=int, default=2, help="number of conv layers" ### changed from 3 to 2
    )
    parser.add_argument(
        "--dropout", type=float, default=0.3, help="dropout ratio" ## changed from 0.0
    )
    parser.add_argument(
        "--lamb", type=float, default=1.0, help="trade-off parameter"
    )
    parser.add_argument(
        "--epochs", type=int, default=500, help="max number of training epochs" ## changed from 1000
    )
    parser.add_argument(
        "--patience", type=int, default=50, help="patience for early stopping" ## changed from 100
    )
    parser.add_argument(
        "--device", type=int, default=-1, help="device id, -1 for cpu"
    )
    parser.add_argument(
        "--dataset_path", type=str, default="./dataset", help="path to dataset"
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=1,
        help="print trainlog every k epochs, -1 for silent training", ### changed from 10
    )
    parser.add_argument(
        "--num_trials", type=int, default=1, help="number of trials"
    )
    parser.add_argument("--output_path", type=str, default="./output")

    args = parser.parse_args()

    # Print statement for debugging
    print("Checking device availability...")

    # device
    args.device = "cpu" if args.device == -1 else "cuda:{}".format(args.device)
    
    ### MY addition
    if not torch.cuda.is_available():
        logging.warning("CUDA is not available, using CPU for training.")
        args.device = "cpu"
    else:
        # Print available devices
        num_devices = torch.cuda.device_count()
        logging.info(f"CUDA is available. Number of devices: {num_devices}")
        print(f"CUDA is available. Number of devices: {num_devices}")

        # Try connecting to the specific device
        try:
            torch.cuda.set_device(int(args.device.split(":")[-1]))
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            logging.info(f"Using GPU device {current_device}: {device_name}")
            print(f"Using GPU device {current_device}: {device_name}")
        except Exception as e:
            logging.error(f"Failed to connect to GPU {args.device}: {e}")
            args.device = "cpu"
    ####

    # print every
    if args.print_every == -1:
        args.print_every = args.epochs + 1

    # bool args
    if args.sample.lower() == "true":
        args.sample = True
    else:
        args.sample = False

    # paths
    if not os.path.exists(args.dataset_path):
        os.makedirs(args.dataset_path)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    name = (
        "Data={}_Hidden={}_Pool={}_WeightDecay={}_Lr={}_Sample={}.log".format(
            args.dataset,
            args.hid_dim,
            args.pool_ratio,
            args.weight_decay,
            args.lr,
            args.sample,
        )
    )
    args.output_path = os.path.join(args.output_path, name)

    return args


def train(model: torch.nn.Module, optimizer, trainloader, device):
    model.train()
    total_loss = 0.0
    num_batches = len(trainloader)
    for batch in trainloader:
        optimizer.zero_grad()
        batch_graphs, batch_labels = batch
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.long().to(device)
        out = model(batch_graphs, batch_graphs.ndata["feat"])
        loss = F.nll_loss(out, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / num_batches


@torch.no_grad()
def test(model: torch.nn.Module, loader, device):
    model.eval()
    correct = 0.0
    loss = 0.0
    num_graphs = 0
    for batch in loader:
        batch_graphs, batch_labels = batch
        num_graphs += batch_labels.size(0)
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.long().to(device)
        out = model(batch_graphs, batch_graphs.ndata["feat"])
        pred = out.argmax(dim=1)
        loss += F.nll_loss(out, batch_labels, reduction="sum").item()
        correct += pred.eq(batch_labels).sum().item()
    return correct / num_graphs, loss / num_graphs


##### My functions: ######
class CustomDGLDataset(Dataset):
    def __init__(self, graph_list):
        self.graphs = graph_list

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        return graph, graph.y


# Function to load graphs from pkl files
def load_custom_graphs(directory, data_type='ovarian', self_loops=True):
    dgl_graph_list = []
    # controls the patients type used in experiment
    invalid_files = ["fp"]
    total_nodes = 0
    total_edges = 0
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
            if self_loops:
                dgl_graph = dgl.add_self_loop(dgl_graph)
            embeddings = [torch.tensor(G.nodes[n]['embedding'], dtype=torch.float32) for n in G.nodes()]
            dgl_graph.ndata['feat'] = torch.stack(embeddings).float() ##  changed from attr to feat (as used in this model)
            dgl_graph.y = torch.tensor(label, dtype=torch.long).squeeze()  # Ensure labels are 1D
            dgl_graph.name = filename
            dgl_graph_list.append(dgl_graph)

            total_nodes += dgl_graph.number_of_nodes()
            total_edges += dgl_graph.number_of_edges()
            if len(dgl_graph_list) > 10:
                break
    print(f"avg number of nodes: {total_nodes / len(dgl_graph_list)}\navg number of edges: {total_edges / len(dgl_graph_list)}")
    return dgl_graph_list

def split_dataset(num_graphs, train_ratio=0.8):
    indices = list(range(num_graphs))
    split = int(train_ratio * num_graphs)
    train_idx, val_idx = indices[:split], indices[split:]
    return train_idx, val_idx

###############

def main(args):
    print("Starting main")
    # ############ ORIGINAL STEP 1: ###########
    # # Step 1: Prepare graph data and retrieve train/validation/test index ============================= #
    # dataset = LegacyTUDataset(args.dataset, raw_dir=args.dataset_path)

    # # add self loop. We add self loop for each graph here since the function "add_self_loop" does not
    # # support batch graph.
    # for i in range(len(dataset)):
    #     dataset.graph_lists[i] = dgl.add_self_loop(dataset.graph_lists[i])

    # num_training = int(len(dataset) * 0.8)
    # num_val = int(len(dataset) * 0.1)
    # num_test = len(dataset) - num_val - num_training
    # train_set, val_set, test_set = random_split(
    #     dataset, [num_training, num_val, num_test]
    # )

    # train_loader = GraphDataLoader(
    #     train_set, batch_size=args.batch_size, shuffle=True, num_workers=6
    # )
    # val_loader = GraphDataLoader(
    #     val_set, batch_size=args.batch_size, num_workers=2
    # )
    # test_loader = GraphDataLoader(
    #     test_set, batch_size=args.batch_size, num_workers=2
    # )

    # device = torch.device(args.device)

    ############ MY STEP 1 (loading data) ##########
    # Step 1: Load custom graph data
    directory = '/dsi/sbm/OrrBavly/colon_data/embedding_graphs_90th_perc_alpha/' 
    data_type = 'colon'
    print("Loading graphs")
    graph_list = load_custom_graphs(directory, data_type)
    dataset = CustomDGLDataset(graph_list)
    print("Finished loading graphs")
    ## add self loops (already adding in load_custom_graphs)
    # for i in range(len(dataset)):
    #     dataset.graphs[i] = dgl.add_self_loop(dataset.graphs[i])

    train_ratio = 0.8
    val_ratio = 0.1
    num_training = int(len(dataset) * train_ratio)
    num_val = int(len(dataset) * val_ratio)
    num_test = len(dataset) - num_val - num_training

    # Use random_split to divide the dataset
    train_set, val_set, test_set = random_split(dataset, [num_training, num_val, num_test])

    #Create DGL data loaders
    train_loader = GraphDataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=6)
    val_loader = GraphDataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = GraphDataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Step 5: Set device
    device = torch.device(args.device)

    # Step 2: Create model =================================================================== #
    #  num_feature, num_classes, _ = dataset.statistics()

        # Extract features and labels for the model
    num_classes = 2  # Assuming binary classification (0 and 1)
    num_feature = dataset[0][0].ndata['feat'].shape[1]


    model = HGPSLModel(
        in_feat=num_feature,
        out_feat=num_classes,
        hid_feat=args.hid_dim,
        conv_layers=args.conv_layers,
        dropout=args.dropout,
        pool_ratio=args.pool_ratio,
        lamb=args.lamb,
        sample=args.sample,
    ).to(device)
    args.num_feature = int(num_feature)
    args.num_classes = int(num_classes)

    # Step 3: Create training components ===================================================== #
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Step 4: training epoches =============================================================== #
    bad_cound = 0
    best_val_loss = float("inf")
    final_test_acc = 0.0
    best_epoch = 0
    train_times = []
    print("Starting loop")
    for e in range(args.epochs):
        s_time = time()
        print("training")
        train_loss = train(model, optimizer, train_loader, device)
        train_times.append(time() - s_time)
        print("Test")
        val_acc, val_loss = test(model, val_loader, device)
        test_acc, _ = test(model, test_loader, device)
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            final_test_acc = test_acc
            bad_cound = 0
            best_epoch = e + 1
        else:
            bad_cound += 1
        if bad_cound >= args.patience:
            break

        # if (e + 1) % args.print_every == 0:
        #     log_format = (
        #         "Epoch {}: loss={:.4f}, val_acc={:.4f}, final_test_acc={:.4f}"
        #     )
        #     print(log_format.format(e + 1, train_loss, val_acc, final_test_acc))
        log_format = (
                "Epoch {}: loss={:.4f}, val_acc={:.4f}, final_test_acc={:.4f}"
            )
        print(log_format.format(e + 1, train_loss, val_acc, final_test_acc))
    
    print(
        "Best Epoch {}, final test acc {:.4f}".format(
            best_epoch, final_test_acc
        )
    )
    return final_test_acc, sum(train_times) / len(train_times)


if __name__ == "__main__":
    args = parse_args()
    res = []
    train_times = []
    for i in range(args.num_trials):
        print("Trial {}/{}".format(i + 1, args.num_trials))
        acc, train_time = main(args)
        res.append(acc)
        train_times.append(train_time)

    mean, err_bd = get_stats(res, conf_interval=False)
    print("mean acc: {:.4f}, error bound: {:.4f}".format(mean, err_bd))

    out_dict = {
        "hyper-parameters": vars(args),
        "result": "{:.4f}(+-{:.4f})".format(mean, err_bd),
        "train_time": "{:.4f}".format(sum(train_times) / len(train_times)),
    }

    with open(args.output_path, "w") as f:
        json.dump(out_dict, f, sort_keys=True, indent=4)