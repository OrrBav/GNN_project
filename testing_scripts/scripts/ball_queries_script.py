import numpy as np
import pandas as pd
import os
import torch
from torch_cluster import radius
import time
from scipy.spatial.distance import pdist, squareform
import scipy.sparse as sp
import json

METRIC = 'cosine'
# cosine / euclidean
PERCENTILES = list(range(1,101,3))
# [5, 15, 25, 35, 50, 70, 80, 90, 95]
R_VALUES = [0.45, 0.5, 0.53, 0.6, 0.65, 0.68, 0.72]
OUTPUT_FILE = "/home/dsi/orrbavly/GNN_project/embeddings/kidney_percentiles/perc_ball_19k_cos_every_other_3.json"
EMBEDDINGS_FOLDER = "/dsi/sbm/OrrBavly/kidney_data/downsamples_19789/embeddings/"


def analyze_ball_pdist(file_path):
    df = pd.read_csv(file_path)
    df = pd.read_csv(file_path)
    embeddings = df.iloc[:, 1:].values.astype('float32')
    
    # Compute pairwise distances
    # Compute distances
    distances = pdist(embeddings, metric='cosine')
    if np.isnan(distances).any():
        print(f"Skipping file {file_path}: NaN values in distances.")
        return None

    # Calculate statistics
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    mean_dist = np.mean(distances)
    percentiles = np.percentile(distances, [25, 50, 75, 90, 95, 99])

    return {
        "min_dist": min_dist,
        "max_dist": max_dist,
        "mean_dist": mean_dist,
        "percentiles": percentiles
    }

# Create json ouput file of percentiles
def run_pdist():
    embeddings_folder = EMBEDDINGS_FOLDER
    output_file = OUTPUT_FILE
    files = os.listdir(embeddings_folder)
    all_results = {}
    i = 1
    all_statistics = []

    print(f"Started working on Metric: {METRIC}\nPercentile: {PERCENTILES}")
    for file in files:
        # Construct full file path
        file_path = os.path.join(embeddings_folder, file)
        # Check if the file is a CSV
        if file.endswith('.csv') and 'fp' not in file:
            print(f"working on file:{file}, number {i}")
            stats = analyze_ball_pdist(file_path)
            if stats is not None:  # Only include valid results
                all_statistics.append(stats)
            # Convert NumPy arrays in percentiles_dict to lists, to work with JSON format
            # percentiles_dict_serializable = {k: v.tolist() for k, v in percentiles_data.items()}
            # all_results[file.split(".")[0]] = percentiles_dict_serializable
            print(f"finished working on file:{file}, number {i}")
            i+=1

        # Aggregate statistics across all graphs
    aggregated_stats = {
        "avg_min_dist": np.mean([s["min_dist"] for s in all_statistics]),
        "avg_max_dist": np.mean([s["max_dist"] for s in all_statistics]),
        "avg_mean_dist": np.mean([s["mean_dist"] for s in all_statistics]),
        "avg_percentiles": np.mean([s["percentiles"] for s in all_statistics], axis=0)
    }

    print("Aggregated Statistics:")
    print(f"Average Min Distance: {aggregated_stats['avg_min_dist']}")
    print(f"Average Max Distance: {aggregated_stats['avg_max_dist']}")
    print(f"Average Mean Distance: {aggregated_stats['avg_mean_dist']}")
    print(f"Average Percentiles: {aggregated_stats['avg_percentiles']}")
    avg_percentiles = aggregated_stats["avg_percentiles"]
    r_values = avg_percentiles[[1, 2, 3, 4]]  # Median, 75th, 90th, and 95th percentiles
    print(f"Suggested Radius Values: {r_values}")


def creating_percentiles_faiss(file_path, r_values, distance_metric='cosine'):
    """
    Creates adjacency matrices using FAISS for radius-based search and prints graph statistics.

    Args:
        file_path (str): Path to the CSV file containing embeddings.
        r_values (list): List of radius values to explore.
        distance_metric (str): 'cosine' or 'euclidean'.

    Returns:
        None
    """
    import faiss

    df = pd.read_csv(file_path)
    embeddings = df.iloc[:, 1:].values.astype('float32')

    # Normalize embeddings for cosine similarity
    if distance_metric == 'cosine':
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product (cosine similarity)
    elif distance_metric == 'euclidean':
        index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance (Euclidean)
    
    index.add(embeddings)  # Add embeddings to FAISS index

    for r in r_values:
        print(f"  Analyzing radius: {r}")

        # Perform FAISS range search (radius-based neighbor search)
        lims, D, I = index.range_search(embeddings, r)

        # Create adjacency matrix
        adjacency_matrix = np.zeros((embeddings.shape[0], embeddings.shape[0]), dtype=np.float32)

        # Fill adjacency matrix with distances
        for i in range(len(lims) - 1):
            start, end = lims[i], lims[i+1]
            neighbors = I[start:end]
            distances = D[start:end]
            adjacency_matrix[i, neighbors] = distances

        # Compute graph statistics
        degrees = adjacency_matrix.sum(axis=1)
        avg_degree = np.mean(degrees)
        max_degree = np.max(degrees)
        min_degree = np.min(degrees)
        sparsity = 1 - (degrees.sum() / (embeddings.shape[0] ** 2))

        # Print results in the same format as the original script
        print(f"    Avg Degree: {avg_degree}, Max Degree: {max_degree}, Min Degree: {min_degree}, Sparsity: {sparsity:.4f}")


def normalize_embeddings(embeddings):
    """Normalize embeddings for cosine similarity (float64 for precision)."""
    embeddings = embeddings.astype(np.float64)
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


def create_sparse_adjacency_matrix(embeddings, r, device='cuda', metric='cosine'):
    """
    Create a sparse adjacency matrix using GPU-accelerated radius search with torch.cdist.

    Args:
        embeddings (np.ndarray): Normalized embeddings (num_nodes x num_features).
        r (float): Radius for ball query.
        device (str): 'cuda' for GPU or 'cpu' for CPU processing.
        metric (str): Distance metric ('cosine' or 'euclidean').

    Returns:
        scipy.sparse.csr_matrix: Sparse adjacency matrix.
    """
    num_nodes = embeddings.shape[0]
    
    # Move embeddings to GPU if available. float64 for precision
    embeddings_torch = torch.tensor(embeddings, dtype=torch.float64, device=device)
    
    # Compute pairwise distances using torch.cdist
    if metric == 'cosine':
        embeddings_torch = embeddings_torch / torch.norm(embeddings_torch, dim=1, keepdim=True)  # Normalize for cosine
        dist_matrix = 1 - torch.matmul(embeddings_torch, embeddings_torch.T)  # Cosine distance
    elif metric == 'euclidean':
        dist_matrix = torch.cdist(embeddings_torch, embeddings_torch, p=2)  # p=2 Euclidean, p=1 Manhattan
    else:
        raise ValueError("Unsupported distance metric. Choose 'cosine' or 'euclidean'.")
    
    # Find neighbors within radius `r`
    row_indices, col_indices = torch.where(dist_matrix <= r)
    
    # Move data back to CPU for sparse matrix creation
    row_indices = row_indices.cpu().numpy()
    col_indices = col_indices.cpu().numpy()
    values = dist_matrix[row_indices, col_indices].cpu().numpy()  # Store actual distances
    
    # Create sparse adjacency matrix
    adj_matrix = sp.csr_matrix((values, (row_indices, col_indices)), shape=(num_nodes, num_nodes))
    
    return adj_matrix

def process_file(file_path, r_values, device='cuda'):
    """Process a single file and generate adjacency matrices & statistics using GPU."""
    print(f"Processing file: {file_path}")
    
    df = pd.read_csv(file_path)
    embeddings = df.iloc[:, 1:].values.astype(np.float64)
    percentiles_dict = {}  # Dictionary to store percentiles

    for r in r_values:
        print(f"  Analyzing radius: {r}")
        # Create adjacency matrix using GPU acceleration
        adj_matrix = create_sparse_adjacency_matrix(embeddings, r, device=device, metric=METRIC)
        distances_flat = adj_matrix.data  # Extract non-zero distances directly
        if len(distances_flat) > 0:  # Ensure there are non-zero distances
            calculated_percentiles = np.percentile(distances_flat, PERCENTILES)
            percentiles_dict[r] = calculated_percentiles
            print(f"    Percentiles: {calculated_percentiles}")
        else:
            percentiles_dict[r] = None
            print(f"    No edges found for radius {r}, skipping percentile calculation.")
        # # Calculate graph statistics
        # degrees = np.array(adj_matrix.sum(axis=1)).flatten()
        # avg_degree = np.mean(degrees)
        # max_degree = np.max(degrees)
        # min_degree = np.min(degrees)
        # sparsity = 1 - (degrees.sum() / (num_nodes * num_nodes))
        # results.append({
        #     "file": os.path.basename(file_path),
        #     "radius": r,
        #     "num_nodes": num_nodes,
        #     "avg_degree": avg_degree,
        #     "max_degree": max_degree,
        #     "min_degree": min_degree,
        #     "sparsity": sparsity
        # })

        # print(f"    Avg Degree: {avg_degree}, Max Degree: {max_degree}, Min Degree: {min_degree}, Sparsity: {sparsity:.4f}")
    return percentiles_dict


def analyze_files(embedding_folder, r_values, device='cuda'):
    """Process multiple embedding files using GPU acceleration."""
    files = [os.path.join(embedding_folder, f) for f in os.listdir(embedding_folder) if f.endswith('.csv')]
    # files_to_analyze = ["12_nd_A_B_H.csv","23_A_B_H.csv","22_nd_A_B_H.csv","11_nd_A_B_OC.csv" ]
    all_results = {}
    print(f"Started working on Metric: {METRIC}\nPercentile: {PERCENTILES}")
    i = 1
    for file in files:
        if 'fp' not in file:
            percentiles_data= process_file(file, r_values, device=device)
            # Convert NumPy arrays in percentiles_dict to lists, to work with JSON format
            percentiles_dict_serializable = {k: v.tolist() for k, v in percentiles_data.items()}
            filename = os.path.basename(file).rsplit(".", 1)[0]  # Remove extension safely. results example: 12_nd_A_B_H
            all_results[filename] = percentiles_dict_serializable
            print(f"finished working on file:{file}, number {i}")
            i+=1
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_results, f, indent=4)  # indent=4 for better readability
    
    print(f"Results saved to {OUTPUT_FILE}")


def gpu_cdist():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("CUDA is not available, using CPU for training.")
    else:
        # Print available devices
        num_devices = torch.cuda.device_count()
        print(f"CUDA is available. Number of devices: {num_devices}")

        # Try connecting to the specific device
        try:
            torch.cuda.set_device(1)  # SET GPU INDEX HERE:
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            print(f"Using GPU device {current_device}: {device_name}")
        except Exception as e:
            device = torch.device("cpu")

    embedding_folder = EMBEDDINGS_FOLDER  # Replace with actual path
    r_values = R_VALUES  # Radius values to test
    analyze_files(embedding_folder, r_values, device=device)  # Use GPU


if __name__ == '__main__':
    start = time.time()
    gpu_cdist()
    # run_percentiles()
    # analyze_ball("/dsi/sbm/OrrBavly/ovarian_data/embeddings/1_A_B_OC.csv")
    end = time.time()
    runtime = end - start
    minutes = int(runtime // 60)
    seconds = int(runtime % 60)
    print(f"Runtime: {minutes} minutes and {seconds} seconds")