## uses dl env 
import pandas as pd
import numpy as np
import faiss
import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import json
import time
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
import gc
import re

import warnings
warnings.filterwarnings("ignore")  # Suppress all warnings 

# json file names and their threshold list:
# results 1(just resulst as filename):[25, 50, 75, 90, 95, 99]
# results 2: [10, 25, 40, 50, 60, 75, 90, 95]
# results 3: [5, 15, 25, 35, 50, 70, 80, 90, 95]
# results 4: [5, 15, 25, 35, 50, 60, 70, 75, 80, 90, 95]
# k_values = [5, 10, 15, 20, int(np.sqrt(N)), int((np.sqrt(N))/2), log2(N)]
# k_values = [5, 10, 20, 50, 100, int(np.sqrt(N)), int((np.sqrt(N))/2), log2(N)]


# Generate percentiles: [1, 5, 10, 15, ..., 95, 99]
PERCENTILES = [1] + list(range(5, 100, 5)) + [99]
# cosine, euclidean (=l2)
METRIC = "cosine"
OUTPUT_FILE = "/home/dsi/orrbavly/GNN_project/embeddings/colon_percentiles/TRB/percentiles_results_cos_every5.json"
# used for output of percentiles OR pottential input for creating netx graphs.
EMBEDDINGS_FOLDER = "/dsi/sbm/OrrBavly/colon_data/embeddings/TRB"

##### NEW COLON #####
NEW_COLON = False
colon_meta_file = "/home/dsi/orrbavly/GNN_project/data/metadata/colon_meta.csv"

##### GPU ####
IF_GPU = False
# OUTPUT_FILE = "/mnt/embeddings/corona_percentiles/perc_results_cos_3_2936.json"
# EMBEDDINGS_FOLDER = "/mnt/corona"
# Set true if you want to include PCA when creating percentiles
RUN_PCA = False
# Globals for Faiss GPU inside docker 
GPU_INDEX = 1
NUM_GPUS = faiss.get_num_gpus()
BATCH_SIZE = 10000
SAVE_INTERVAL = 5
CHECKPOIN_DIR = "/mnt/embeddings/corona_percentiles/checkpoints/"
PROCESSED_FILES_LOG = os.path.join(CHECKPOIN_DIR, "processed_files.txt")
print(CHECKPOIN_DIR)
print(PROCESSED_FILES_LOG)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"



def tag_colon_by_metadata(file, sample_n_map):
    """
    Extracts sample_id from the filename, looks up its N value in metadata,
    and returns a tagged filename (e.g., 'P1-S1_high') or None if skipped.
    """
    # samples with wrong medical annotations, should be ignored.
    invalid_files = [
    'P1-S24', 'P2-S22', 'P3-S3', 'P3-S22', 'P4-S4',
    'P5-S20', 'P6-S12', 'P6-S22', 'P7-S22', 'P9-S8', 'P9-S24'
    ]
       # Extract sample ID (e.g., P1-S10) from filename
    match = re.search(r'(P\d+-S\d+)', file)
    if not match:
        print(f"Skipping {file} (no sample ID found)")
        return None

    sample_id = match.group(1)

    # Skip if sample is in the invalid list
    if sample_id in invalid_files:
        print(f"Skipping {file} (marked as invalid - wrong annotation)")
        return None

    # Check if sample ID is in metadata
    if sample_id not in sample_n_map:
        print(f"Skipping {file} (sample ID not found in metadata)")
        return None

    n_value = sample_n_map[sample_id]

    # Determine tag based on N
    if n_value == '0':
        tag = "_low"
    elif n_value in ['1', '2']:
        tag = "_high"
    else:
        print(f"Skipping {file} (unexpected N value: {n_value})")
        return None

    # Return modified filename (without .csv)
    base_name = file.split(".")[0]
    return base_name + tag

def apply_pca_and_run_algorithm(file_path, explained_variance=0.95):
    # Load the data
    df = pd.read_csv(file_path)
    tcr_sequences = df.iloc[:, 0].values
    embeddings = df.iloc[:, 1:].values.astype('float32')

    # # Apply PCA
    # print(f"Original embedding shape: {embeddings.shape}")
    # pca = PCA(n_components=explained_variance)  # Retain 95% variance
    # # pca = PCA(n_components=min(embeddings.shape[0], 100))  # Or other values
    # reduced_embeddings = pca.fit_transform(embeddings)
    # print(f"Reduced embedding shape: {reduced_embeddings.shape}")
    
    # Replace embeddings in your existing algorithm
    def create_faiss_index(embeddings, k, distance_metric='cosine'):
        if distance_metric == 'cosine':
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
            index = faiss.IndexFlatIP(embeddings.shape[1])
        elif distance_metric == 'euclidean':
            index = faiss.IndexFlatL2(embeddings.shape[1])
        else:
            raise ValueError("Invalid Distance Metric value")
        index.add(embeddings)
        distances, indices = index.search(embeddings, k)
        return distances, indices

    def create_adjacency_matrix(num_embeddings, indices, distances, k):
        adjacency_matrix = np.zeros((num_embeddings, num_embeddings))
        for i in range(num_embeddings):
            for j in range(1, k):  # Skip the first neighbor (itself)
                adjacency_matrix[i, indices[i, j]] = distances[i, j]
        return adjacency_matrix

    # Define different k values to explore
    N = embeddings.shape[0] ## change to reduced_embedding when running PCA
    k_values = [5, 10, 15, 20, 50, int(np.sqrt(N)), int(np.sqrt(N)/2)]
    log_k = int(np.log(N)) + 1
    if log_k not in k_values:
        k_values.append(log_k)

    percentiles_dict = {}

    for k in k_values:
        # FAISS distances/indices using original embeddings
        distances, indices = create_faiss_index(embeddings, k, distance_metric=METRIC)
        adjacency_matrix = create_adjacency_matrix(embeddings.shape[0], indices, distances, k)
        # Flatten adjacency matrix and filter non-zero values
        distances_flat = adjacency_matrix.flatten()
        distances_flat = distances_flat[distances_flat > 0]
        original_percentiles  = np.percentile(distances_flat, PERCENTILES)
        
        # Add Spectral Embedding Calculation
        # Convert adjacency matrix to spectral embeddings
        spectral_embedder = SpectralEmbedding(n_components=50, affinity='precomputed')
        spectral_embeddings = spectral_embedder.fit_transform(adjacency_matrix)
        # Compute FAISS distances/indices using spectral embeddings
        spectral_distances, spectral_indices = create_faiss_index(spectral_embeddings, k)
        spectral_adj_matrix = create_adjacency_matrix(N, spectral_indices, spectral_distances, k)
        # Flatten spectral adjacency matrix and filter non-zero values
        spectral_flat = spectral_adj_matrix.flatten()
        spectral_flat = spectral_flat[spectral_flat > 0]
        spectral_percentiles = np.percentile(spectral_flat, PERCENTILES)


        # # Store percentiles for original and spectral embeddings
        # percentiles_dict[k] = {
        #     "original": original_percentiles.tolist(),
        #     "spectral": spectral_percentiles.tolist(),
        #     "combined": np.hstack((original_percentiles, spectral_percentiles)).tolist()
        # }
        
        percentiles_dict[k] = np.hstack((original_percentiles, spectral_percentiles))
        # Optionally print progress
        # print(f"Percentiles for k={k}: Original={original_percentiles}, Spectral={spectral_percentiles}")

    return percentiles_dict


def creating_percentiles_gpu(file_path):
    import torch

    df = pd.read_csv(file_path)
    # Extract TCR sequences and embeddings
    tcr_sequences = df.iloc[:, 0].values
    embeddings = df.iloc[:, 1:].values.astype('float32')

    # Function to create FAISS index and search for nearest neighbors
    def create_faiss_index(embeddings, k, distance_metric='cosine'):
        if IF_GPU:
            res = faiss.StandardGpuResources()
            res.setDefaultNullStreamAllDevices()  # Enable multi-streaming for better parallelism

        if distance_metric == 'cosine':
            # Ensure embeddings are normalized properly
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            index = faiss.GpuIndexFlatIP(res, embeddings.shape[1])  # Cosine similarity (Inner Product)
        
        elif distance_metric == 'euclidean':
            index = faiss.GpuIndexFlatL2(res, embeddings.shape[1])  # Euclidean distance

        index.add(embeddings)
        return index

    # Function to create the adjacency matrix
    def create_adjacency_matrix(num_embeddings, indices, distances, k):
        adjacency_matrix = np.zeros((num_embeddings, num_embeddings), dtype=np.float32)
        for i in range(num_embeddings):
            for j in range(1, k):  # Skip the first neighbor (itself)
                adjacency_matrix[i, indices[i, j]] = distances[i, j]
        return adjacency_matrix

    def batch_faiss_search(index, embeddings, k, batch_size=BATCH_SIZE):
        distances_list, indices_list = [], []
        # Ensure embeddings are normalized before searching (to match embeddings in Index)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        for i in range(0, embeddings.shape[0], batch_size):
            batch = embeddings[i:i + batch_size]
            distances, indices = index.search(batch, k)
            distances_list.append(distances)
            indices_list.append(indices)
        
        return np.vstack(distances_list), np.vstack(indices_list)

    # Define different k values to explore
    N = embeddings.shape[0]
    log_k = int(np.log(N))
    if log_k == 0:
        log_k += 1
    k_values = [5, 10, 15, 20, int(np.sqrt(N)), int((np.sqrt(N))/2)]
    if log_k in k_values:
        log_k += 1
    k_values.append(log_k)

    percentiles = PERCENTILES  # Standard percentiles to calculate
    percentiles_dict = {}  # Dictionary to store percentiles

    for k in k_values:
        # Run for single GPU
        # Clear memory before starting
        torch.cuda.empty_cache()
        gc.collect()
        index = create_faiss_index(embeddings, k, distance_metric=METRIC)
        index.nprobe = 10  # Use multiple probes for more accurate results
        distances, indices = batch_faiss_search(index, embeddings, k)
        
        adjacency_matrix = create_adjacency_matrix(embeddings.shape[0], indices, distances, k)

        distances_flat = adjacency_matrix.flatten()
        distances_flat = distances_flat[distances_flat > 0]
        calculated_percentiles = np.percentile(distances_flat, percentiles)
        # print(f"[DEBUG] Percentiles for k={k}: {calculated_percentiles}")
        percentiles_dict[k] = calculated_percentiles

        # Ensure FAISS is fully reset
        del index
        res = None
        gc.collect()
        torch.cuda.empty_cache()

    return percentiles_dict


def load_processed_files():
    """Load already processed file names to avoid reprocessing."""
    if os.path.exists(PROCESSED_FILES_LOG):
        with open(PROCESSED_FILES_LOG, "r") as f:
            processed_files = f.read().splitlines()
        return set(processed_files), len(processed_files)
    return set(), 0

def save_checkpoint(all_results, processed_files, start_idx, end_idx):
    """Save current percentiles data and update processed files log."""
    # Ensure the directory exists
    os.makedirs(CHECKPOIN_DIR, exist_ok=True)
    json_filename = os.path.join(CHECKPOIN_DIR, f"perc_faiss_cos_every5_{start_idx}-{end_idx}.json")
    print(f"Saving interval: {start_idx}-{end_idx}")
    # Save results as JSON
    with open(json_filename, "w") as json_file:
        json.dump(all_results, json_file, indent=4)
    
    # Append processed files to log
    with open(PROCESSED_FILES_LOG, "a") as f:
        for file in processed_files:
            f.write(file + "\n")

def run_percentiles_gpu():
    import torch
    torch.cuda.set_device(GPU_INDEX)
    print(f"[INFO] PyTorch default CUDA device: {torch.cuda.current_device()}")

    embeddings_folder = EMBEDDINGS_FOLDER
    output_file = OUTPUT_FILE
    files = os.listdir(embeddings_folder)
    processed_files, last_processed_count = load_processed_files()  # Load processed files to skip them
    all_results = {}
    batch_processed_files = []
    i = last_processed_count + 1
    start_idx = i  # Track the start index of the batch (according to procceced_files file.)
    print(f"Started working on Metric: {METRIC}\nPercentile: {PERCENTILES}")
    print(f"Running PCA:\t{RUN_PCA}")
    print(f"Running on GPU:\t{IF_GPU}")
    print(f"Saving each {SAVE_INTERVAL} interval")
    for file in files:
        file_path = os.path.join(embeddings_folder, file)
        # Skip non-CSV files and already processed files
        if not file.endswith('.csv') or 'fp' in file or file in processed_files:
            continue
        print(f"Working on file: {file}, number {i}")
        start = time.time()

        try:
            # Run PCA or FAISS-based algorithm
            if RUN_PCA:
                percentiles_data = apply_pca_and_run_algorithm(file_path)
            else:
                percentiles_data = creating_percentiles_gpu(file_path)

            # Convert NumPy arrays to lists for JSON serialization
            percentiles_dict_serializable = {k: v.tolist() for k, v in percentiles_data.items()}
            all_results[file.split(".")[0]] = percentiles_dict_serializable
            batch_processed_files.append(file)

            print(f"Finished working on file: {file}, number {i}")
            end = time.time()
            print(f"Time processing sample: {(end - start) / 60:.1f} minutes and {(end - start) % 60:.1f} seconds")
        
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue  # Continue with the next file even if one fails

        # Save every SAVE_INTERVAL files
        if (i - 1) % SAVE_INTERVAL == 0 and i > last_processed_count:
            end_idx = i   # Determine end index
            save_checkpoint(all_results, batch_processed_files, start_idx, end_idx)
            all_results.clear()  # Clear dict to free memory
            batch_processed_files.clear()  # Clear processed file list
            start_idx = i  # Update start index for next batch
        
        i += 1

    # Save remaining data if any
    if all_results:
        end_idx = i - 1
        save_checkpoint(all_results, batch_processed_files, start_idx, end_idx)

    print("Processing complete! All results saved.")

def creating_percentiles(file_path):
    df = pd.read_csv(file_path)
    # Extract TCR sequences and embeddings
    tcr_sequences = df.iloc[:, 0].values
    embeddings = df.iloc[:, 1:].values.astype('float32')

    # Function to create FAISS index and search for nearest neighbors
    def create_faiss_index(embeddings, k, distance_metric = 'cosine'):
        if distance_metric == 'cosine':
            # Normalize the embeddings
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
            # When embeddings are normalized, the inner product is equivalent to cosine similarity. 
            index = faiss.IndexFlatIP(embeddings.shape[1])  # Using inner product(IP) for cosine similarity
        elif distance_metric == 'euclidean':
            # should not normelize beforeahand, or it will distort their original magnitudes
            index = faiss.IndexFlatL2(embeddings.shape[1])  # Using L2 distance (Euclidean)
        index.add(embeddings)
        distances, indices = index.search(embeddings, k)
        return distances, indices

    # Function to create the adjacency matrix
    def create_adjacency_matrix(num_embeddings, indices, distances, k):
        adjacency_matrix = np.zeros((num_embeddings, num_embeddings))
        for i in range(num_embeddings):
            for j in range(1, k):  # Skip the first neighbor (itself)
                adjacency_matrix[i, indices[i, j]] = distances[i, j]
        return adjacency_matrix

    # Define different k values to explore
    N = embeddings.shape[0]
    log_k = int(np.log(N))
    if log_k == 0:
        log_k += 1
    k_values = [5, 10, 15, 20, int(np.sqrt(N)), int((np.sqrt(N))/2)]
    # Ensure log_k is unique
    if log_k in k_values:
        log_k += 1
    # Add log_k to k_values
    k_values.append(log_k)
    # Threshold:
    percentiles = PERCENTILES # Standard percentiles to calculate

    percentiles_dict = {}  # Dictionary to store percentiles

    for k in k_values:
        distances, indices = create_faiss_index(embeddings, k, distance_metric=METRIC)
        adjacency_matrix = create_adjacency_matrix(embeddings.shape[0], indices, distances, k)

        ## TODO: remove below code for graph stats(made to compare to ball radius)    
        # # Compute graph statistics
        # degrees = adjacency_matrix.sum(axis=1)
        # avg_degree = np.mean(degrees)
        # max_degree = np.max(degrees)
        # min_degree = np.min(degrees)
        # sparsity = 1 - (degrees.sum() / (embeddings.shape[0] ** 2))

        # # Print results in the same format as the original script
        # print(f"K value: {k}    Avg Degree: {avg_degree}, Max Degree: {max_degree}, Min Degree: {min_degree}, Sparsity: {sparsity:.4f}")

        distances_flat = adjacency_matrix.flatten()
        distances_flat = distances_flat[distances_flat > 0]
        calculated_percentiles = np.percentile(distances_flat, percentiles)
        percentiles_dict[k] = calculated_percentiles

        # # Print the percentiles for each k value
        print(f"Percentiles of Distances for k={k}: {calculated_percentiles}")

    return percentiles_dict

# Create json ouput file of percentiles
def run_percentiles():
    embeddings_folder = EMBEDDINGS_FOLDER
    output_file = OUTPUT_FILE
    files = os.listdir(embeddings_folder)
    all_results = {}
    i = 1
    print(f"Started working on Metric: {METRIC}\nPercentile: {PERCENTILES}")
    print(f"Running PCA:\t{RUN_PCA}")
    print(f"Running on GPU:\t{IF_GPU}")
    for file in files:
        # Construct full file path
        file_path = os.path.join(embeddings_folder, file)
        # Check if the file is a CSV
        if file.endswith('.csv') and 'fp' not in file:
            print(f"working on file:{file}, number {i}")
            filename = file.split(".")[0]
            if NEW_COLON:
                # Load metadata and create sample_id → N map
                df_meta = pd.read_csv(colon_meta_file)
                sample_n_map = dict(zip(df_meta["sample_id"], df_meta["N"]))
                filename = tag_colon_by_metadata(file, sample_n_map)
                if filename is None:
                    continue
            start = time.time()
            if RUN_PCA:
                percentiles_data = apply_pca_and_run_algorithm(file_path) ### TODO: change back to creating_percentiles
            elif IF_GPU:
                percentiles_data = creating_percentiles_gpu(file_path)
            else:
                percentiles_data = creating_percentiles(file_path)
            # Convert NumPy arrays in percentiles_dict to lists, to work with JSON format
            percentiles_dict_serializable = {k: v.tolist() for k, v in percentiles_data.items()}
            all_results[filename] = percentiles_dict_serializable
            print(f"finished working on file:{file}, number {i}")
            i+=1
            end = time.time()
            print(f"Time proccesing sample: {(end - start)//60} minutes and {(end - start) % 60} seconds" )

    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=4)  # indent=4 for better readability
    
    print(f"Results saved to {output_file}")

def create_csvs():
    # Function to create csv files for each patient from the combined patients data csv (named: "data_all_ab")
    print("Reading data csv")
    tcrb_data = pd.read_csv("/home/users/orrbavly/GNN_project/data_all_ab.csv")
    print("Finished loading data csv")
    # Assuming tcrb_data and OC_df are already defined
    H_df = tcrb_data[tcrb_data.iloc[:, 0].str.endswith("_H")]
    sampled_H_df = H_df.sample(n=34, random_state=42)

    # Get the indices of the sampled rows
    sampled_indices = sampled_H_df.index

    # Get the rows that are not in the sampled indices
    remaining_H_df = H_df.drop(sampled_indices)
    
    non_zero_counts = remaining_H_df.astype(bool).sum()
    filtered_columns = non_zero_counts[non_zero_counts >= 1].index
    filtered_df = remaining_H_df[filtered_columns]

    for index, row in filtered_df.iterrows():
        non_zero_count = (row != 0).sum()
        print(f"Row {index} has {non_zero_count} non-zero cells")

    filtered_df.rename(columns={"Unnamed: 0": "Sample"}, inplace=True)

    results = {}
    # create dict, later save it as csv
    for row in filtered_df.itertuples(index=False, name='Row'):
        sample_name = row.Sample
        non_zero_columns = [filtered_df.columns[i] for i, value in enumerate(row[1:], start=1) if value != 0]
        results[sample_name] = non_zero_columns
    
    # create csv files
    for sample, column in results.items():
        csv_df = pd.DataFrame(column)
        csv_df.columns = ['Sequences']
        parts = sample.split("_")
        output_path = f"/home/users/orrbavly/GNN_project/embeddings/csvs/extras/{parts[0]}_{parts[-1]}.csv"
        csv_df.to_csv(output_path, index=False)
        print(f"finished with: {parts[0]}_{(parts[-1])}")

#######################################################################
##### Creating networkx graphs with embeddings from data, and saving them
#######################################################################

def filter_adjacency_matrix(adjacency_matrix, threshold):
    """
    Filters an adjacency matrix by setting values below a threshold to zero.
    Parameters:
    - adjacency_matrix (np.ndarray): The original adjacency matrix.
    - threshold (float): The threshold value. Only values greater than this threshold will be kept.
    Returns:
    - filtered_matrix (np.ndarray): The filtered adjacency matrix.
    """
    # Copy the original adjacency matrix to avoid modifying it
    filtered_matrix = np.copy(adjacency_matrix)
    # Apply the threshold filter
    filtered_matrix[filtered_matrix < threshold] = 0
    
    return filtered_matrix

def create_netx_graphs(file_path):
    df = pd.read_csv(file_path)
    # Extract TCR sequences and embeddings
    tcr_sequences = df.iloc[:, 0].values
    embeddings = df.iloc[:, 1:].values.astype('float32')

    # Function to create FAISS index and search for nearest neighbors
    def create_faiss_index(embeddings, k, distance_metric = 'cosine'):
        if distance_metric == 'cosine':
            # Normalize the embeddings
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
            # When embeddings are normalized, the inner product is equivalent to cosine similarity. 
            index = faiss.IndexFlatIP(embeddings.shape[1])  # Using inner product(IP) for cosine similarity
        elif distance_metric == 'euclidean':
            # should not normelize beforeahand, or it will distort their original magnitudes
            index = faiss.IndexFlatL2(embeddings.shape[1])  # Using L2 distance (Euclidean)
        index.add(embeddings)
        distances, indices = index.search(embeddings, k)
        return distances, indices

    # Function to create the adjacency matrix
    def create_adjacency_matrix(num_embeddings, indices, distances, k, threshold=None):
        adjacency_matrix = np.zeros((num_embeddings, num_embeddings))
        for i in range(num_embeddings):
            for j in range(1, k):  # Skip the first neighbor (itself)
                adjacency_matrix[i, indices[i, j]] = distances[i, j]
        return adjacency_matrix

    # Define different k values to explore
    N = embeddings.shape[0]
    k_value = int(np.sqrt(N))
    percentiles = [25, 50, 80, 90, 95, 99]
    # Threshold:

    distances, indices = create_faiss_index(embeddings, k_value, distance_metric=METRIC)
    adjacency_matrix = create_adjacency_matrix(embeddings.shape[0], indices, distances, k_value)
    
    distances_flat = adjacency_matrix.flatten()
    distances_flat = distances_flat[distances_flat > 0]
    calculated_percentiles = np.percentile(distances_flat, percentiles)
    ## AFTER calculating the percentiles, create a new Adjecency matrix, filtered bases on given THRESHOLD

    # Select a threshold based on a desired percentile (e.g., 90th percentile)
    threshold = calculated_percentiles[2]  # 80th percentile
    filtered_adjacency_matrix = filter_adjacency_matrix(adjacency_matrix, threshold)

    # Create an empty graph
    G = nx.Graph()

    # Add nodes with TCR sequence names and embeddings as attributes
    for i, seq in enumerate(tcr_sequences):
        G.add_node(seq, embedding=embeddings[i])

    # Add edges based on the filtered adjacency matrix. weight is the cosine simmilarity
    for i, seq_i in enumerate(tcr_sequences):
        for j, seq_j in enumerate(tcr_sequences):
            if filtered_adjacency_matrix[i, j] > 0:
                G.add_edge(seq_i, seq_j, weight=filtered_adjacency_matrix[i, j])

    return G

def save_netx_graphs(directory):
    # /dsi/sbm/OrrBavly/colon_data/embedding_graphs_90th_perc_alpha/
    output_path_graphs = '/dsi/sbm/OrrBavly/colon_data/embedding_graphs_80th_perc_alpha/'
    # Get a list of already processed files (without their extensions)
    processed_files = {os.path.splitext(filename)[0] for filename in os.listdir(output_path_graphs)}

    for filename in os.listdir(directory):
        # if 'nd' not in filename and 'nh' not in filename:
        # Get the base filename without extension
        base_filename = os.path.splitext(filename)[0]
        if base_filename not in processed_files:
            print(f"Working on file: {filename}")
            file_path = os.path.join(directory, filename)
            graph = create_netx_graphs(file_path)
            local_output_path = os.path.join(output_path_graphs, os.path.splitext(filename)[0])
            # Save the graph using Python's pickle module
            with open(local_output_path, 'wb') as f:
                pickle.dump(graph, f)
            print(f"finished working on file: {filename}")
        else:
            print(f"Skipping file: {base_filename}" )


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


if __name__ == '__main__':
    start = time.time()
    run_percentiles()
    # save_netx_graphs('/dsi/sbm/OrrBavly/colon_data/embeddings/TRA/')
    end = time.time() 
    print(f"Runtime: {(end - start)/60}")
    
