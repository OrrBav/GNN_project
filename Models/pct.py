import torch
import pandas as pd
import os
import torch.nn as nn
import torch.nn.functional as F
import logging
from torch.utils.data import Subset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import time

GPU_DEVICE = 1
DATA_TYPE = 'ovarian'
EMBEDDING_FOLDER = "/home/dsi/orrbavly/GNN_project/embeddings/new_embeddings/romi_embeddings"

def pad_embeddings(embeddings_list):
    # Convert embeddings to tensors and pad
    tensors = [torch.tensor(e, dtype=torch.float32) for e in embeddings_list]
    padded = pad_sequence(tensors, batch_first=True)  # Shape: (num_patients, max_length, feature_dim)
    return padded

def procces_data(embeddings_folder):
    """
    should create embeddings and lables vectors.
    """
    print("Proccesing files")
    embeddings_list = []  
    labels_list = []  
    patients = []  

    for file in os.listdir(embeddings_folder):
        file_path = os.path.join(embeddings_folder, file)
        if file.endswith('.csv') and 'fp' not in file:
            df = pd.read_csv(file_path)
            embeddings = df.iloc[:, 1:].values.astype('float32')  # Extract embeddings
            embeddings_list.append(torch.tensor(embeddings, dtype=torch.float32))
            if DATA_TYPE == 'colon':
                labels_list.append(1 if "high" in file else 0)  # Binary label
            elif DATA_TYPE == 'ovarian':
                labels_list.append(1 if "OC" in file else 0)  # Binary label
            else:
                raise KeyError("invalid Data type value.")
            patients.append(file)

    # Pad sequences to the same length
    embeddings_tensor = pad_sequence(embeddings_list, batch_first=True)  # Shape: (num_patients, max_length, feature_dim)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)  # Shape: (num_patients,)

    return embeddings_tensor, labels_tensor, patients


def split_fold10_manual(labels, fold_idx=0, n_splits=10, random_seed=0):
    """
    Manual implementation for StratifiedKFold function from sklearn, because of conflicting issues when trying to install sklearn in conda env.
    """
    labels = np.array(labels)  # Ensure labels are a NumPy array
    np.random.seed(random_seed)
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

class SimplePCT(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimplePCT, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Transformer layers
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=4),
            num_layers=2
        )
        
        # Classification head
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Initial embedding
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Transformer-based attention
        x = self.transformer(x)
        
        # Sum pooling
        x = torch.sum(x, dim=1)  # Shape: (batch_size, 128)
        
        # Classification head
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=-1)

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        # forward
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # backward pass and optimization
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Calculate training accuracy
        pred = outputs.argmax(dim=1)
        correct += pred.eq(targets).sum().item()
        total_samples += targets.size(0)
    train_accuracy = correct / total_samples
    return total_loss / len(train_loader), train_accuracy


@torch.no_grad()
def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()
        pred = outputs.argmax(dim=1)
        correct += pred.eq(targets).sum().item()
    accuracy = correct / len(test_loader.dataset)
    return total_loss / len(test_loader), accuracy


def set_cuda():
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
    return device

if __name__ == "__main__":
    start = time.time()
    device = set_cuda()
    num_epochs = 150
    n_splits = 10
    batch_size = 16
    num_classes = 2 

    embeddings, labels, patients = procces_data(EMBEDDING_FOLDER)
    dataset = TensorDataset(embeddings, labels) # embeddings is expected: (num_patients, max_length(TCR seq), feature_dim)
    
    num_patients, max_seq_len, feature_dim = embeddings.shape
    
    # create cross-validation samples for train/test
    all_test_accuracies = []
    all_test_losses = []

    for fold_idx in range(n_splits):
        print(f"Fold {fold_idx + 1}/{n_splits}")
        train_idx, test_idx = split_fold10_manual(labels, fold_idx, n_splits)
        train_set = Subset(dataset, train_idx)
        test_set = Subset(dataset, test_idx)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        pct_model = SimplePCT(input_dim=feature_dim, num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(pct_model.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            train_loss, train_accuracy = train(pct_model, train_loader, optimizer, criterion, device)
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        test_loss, test_accuracy = test(pct_model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        all_test_accuracies.append(test_accuracy)
        all_test_losses.append(test_loss)

    # Aggregate results
    mean_test_accuracy = sum(all_test_accuracies) / n_splits
    mean_test_loss = sum(all_test_losses) / n_splits
    print(f"Average Test Loss: {mean_test_loss:.4f}, Average Test Accuracy: {mean_test_accuracy:.4f}")
    end = time.time()
    print(f"Runtime: {(end - start)/60}")