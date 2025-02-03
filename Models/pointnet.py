import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import random, os, time
import pandas as pd

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 150
L_RATE = 0.001
DATA_DIR = "/dsi/sbm/OrrBavly/kidney_data/downsamples_19789/embeddings/"


# Define a simple dataset for TCR sequence embeddings
class TCRDataset(Dataset):
    def __init__(self, data, labels):
        # data: List of tensors (each tensor is a point cloud for a patient)
        # labels: List of binary labels (0 or 1)
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Define the PointNet model
class PointNet(nn.Module):
    def __init__(self, input_dim=768):
        super(PointNet, self).__init__()

        # Input transformation
        self.input_transform = nn.Sequential(
            nn.Conv1d(input_dim, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        # Feature transformation
        self.feature_transform = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [batch_size, num_points, input_dim]
        x = x.transpose(1, 2)  # Change to [batch_size, input_dim, num_points]

        # Input transformation
        x = self.input_transform(x)
        x = torch.max(x, 2)[0]  # Global max pooling

        # Feature transformation
        x = self.feature_transform(x.unsqueeze(-1))
        x = torch.max(x, 2)[0]  # Global max pooling

        # Classification
        x = self.fc(x)
        return x

# Function to load data from files
def load_data(data_dir):
    data = []
    labels = []

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".csv"):
            file_path = os.path.join(data_dir, file_name)
            df = pd.read_csv(file_path)

            # Extract embeddings and convert to tensor
            embeddings = torch.tensor(df.iloc[:, 1:].values, dtype=torch.float32)
            data.append(embeddings)

            # Deduce label from file name
            if "STA" in file_name:
                labels.append(0)
            elif "AR" in file_name:
                labels.append(1)

    return data, labels

# Function to test the model
def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            test_loss += loss.item()

            # Calculate accuracy
            predictions = (outputs.squeeze() > 0.5).long()
            # correct += (predictions == targets).sum().item()
            # total += targets.size(0)
            # Update confusion matrix components
            tp += ((predictions == 1) & (targets == 1)).sum().item()
            tn += ((predictions == 0) & (targets == 0)).sum().item()
            fp += ((predictions == 1) & (targets == 0)).sum().item()
            fn += ((predictions == 0) & (targets == 1)).sum().item()

    # Calculate balanced accuracy
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
    balanced_accuracy = (sensitivity + specificity) / 2

    print(f"Test Loss: {test_loss / len(test_loader):.4f}, Balanced Accuracy: {balanced_accuracy:.4f}")
    # accuracy = correct / total
    # print(f"Test Loss: {test_loss / len(test_loader):.4f}, Accuracy: {accuracy:.4f}")

# Manual train-test split function with stratification
def manual_train_test_split(data, labels, test_size=0.2, random_seed=42):
    random.seed(random_seed)

    # Group data by label
    grouped_data = {0: [], 1: []}
    for d, l in zip(data, labels):
        grouped_data[l].append(d)

    train_data, test_data, train_labels, test_labels = [], [], [], []

    for label, items in grouped_data.items():
        random.shuffle(items)
        split_idx = int(len(items) * (1 - test_size))
        train_data.extend(items[:split_idx])
        test_data.extend(items[split_idx:])
        train_labels.extend([label] * split_idx)
        test_labels.extend([label] * (len(items) - split_idx))

    return train_data, test_data, train_labels, test_labels

# Define a custom collate function for padding
def collate_fn(batch):
    # Separate data and labels
    data, labels = zip(*batch)
    # Pad data tensors to the same size
    padded_data = pad_sequence(data, batch_first=True, padding_value=0.0)
    labels = torch.tensor(labels, dtype=torch.float32)
    return padded_data, labels

def load_gpu():   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("CUDA is not available, using CPU for training.")
    else:
        num_devices = torch.cuda.device_count()
        print(f"CUDA is available. Number of devices: {num_devices}")

        # Try connecting to the specific device
        try:
            torch.cuda.set_device(0)  # SET GPU INDEX HERE:
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            print(f"Using GPU device {current_device}: {device_name}")
        except Exception as e:
            print(f"Failed to connect to GPU: {e}")
            device = torch.device("cpu")

    print(f"Using device: {device}")

if __name__ == '__main__':
    start = time.time()
    load_gpu()

    print("Loading Data...")
    data, labels = load_data(DATA_DIR)

    # Split data into train and test sets manually
    data_train, data_test, labels_train, labels_test = manual_train_test_split(data, labels, test_size=0.2)

    # Create datasets and dataloaders
    train_dataset = TCRDataset(data_train, labels_train)
    test_dataset = TCRDataset(data_test, labels_test)

    # Create dataloaders with the collate function (ensures uniform input dimensions)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # Model, loss, and optimizer
    model = PointNet(input_dim=768)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=L_RATE, weight_decay=1e-4)

    # Training loop
    print("Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {running_loss / len(train_loader):.6f}")

        # Test the model every 5 epochs
        if (epoch + 1) % 5 == 0:
            test_model(model, test_loader, criterion)

    print("Training complete.")
    end = time.time()
    print(f"Total Running time: {(end - start) / 60}")