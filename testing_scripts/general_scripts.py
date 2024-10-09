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

input_folder = "./embeddings/cvcs/"
# json file names and their threshold list:
# results 1(just resulst as filename):[25, 50, 75, 90, 95, 99]
# results 2: [10, 25, 40, 50, 60, 75, 90, 95]
# results 3: [5, 15, 25, 35, 50, 70, 80, 90, 95]
PERCENTILES = [5, 15, 25, 35, 50, 70, 80, 90, 95] ## does not matter when creating netx graphs function
# cosine, euclidean (=l2)
METRIC = "cosine"
OUTPUT_FILE = "/home/dsi/orrbavly/GNN_project/embeddings/colon_percentiles/percentiles_results_check_cos_3_all.json"
EMBEDDINGS_FOLDER = "/dsi/sbm/OrrBavly/colon_data/embeddings"
# add /extras for the additional csvs
input_emb_folder = "/home/dsi/orrbavly/GNN_project/embeddings/new_embeddings/csvs"


def creating_percentiles(file_path):
    df = pd.read_csv(file_path)
    # Extract TCR sequences and embeddings
    tcr_sequences = df.iloc[:, 0].values
    embeddings = df.iloc[:, 1:].values.astype('float32')

    # # Normalize the embeddings
    # embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]

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
        
        distances_flat = adjacency_matrix.flatten()
        distances_flat = distances_flat[distances_flat > 0]
        calculated_percentiles = np.percentile(distances_flat, percentiles)
        percentiles_dict[k] = calculated_percentiles

        # Print the percentiles for each k value
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
    for file in files:
        # Construct full file path
        file_path = os.path.join(embeddings_folder, file)
        # Check if the file is a CSV
        if file.endswith('.csv'):
            print(f"working on file:{file}, number {i}")
            percentiles_data = creating_percentiles(file_path)
            # Convert NumPy arrays in percentiles_dict to lists
            percentiles_dict_serializable = {k: v.tolist() for k, v in percentiles_data.items()}
            all_results[file.split(".")[0]] = percentiles_dict_serializable
            print(f"finished working on file:{file}, number {i}")
            i+=1
    
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
    percentiles = [25, 50, 75, 90, 95, 99]
    # Threshold:

    distances, indices = create_faiss_index(embeddings, k_value, distance_metric=METRIC)
    adjacency_matrix = create_adjacency_matrix(embeddings.shape[0], indices, distances, k_value)
    
    distances_flat = adjacency_matrix.flatten()
    distances_flat = distances_flat[distances_flat > 0]
    calculated_percentiles = np.percentile(distances_flat, percentiles)
    ## AFTER calculating the percentiles, create a new Adjecency matrix, filtered bases on given THRESHOLD

    # Select a threshold based on a desired percentile (e.g., 90th percentile)
    threshold = calculated_percentiles[3]  # 90th percentile
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
    output_path = '/home/dsi/orrbavly/GNN_project/data/embedding_graphs_90th_perc_new'
    # Get a list of already processed files (without their extensions)
    processed_files = {os.path.splitext(filename)[0] for filename in os.listdir(output_path)}

    for filename in os.listdir(directory):
        # if 'nd' not in filename and 'nh' not in filename:
        # Get the base filename without extension
        base_filename = os.path.splitext(filename)[0]
        if base_filename not in processed_files:
            print(f"Working on file: {filename}")
            file_path = os.path.join(directory, filename)
            graph = create_netx_graphs(file_path)
            local_output_path = os.path.join(output_path, os.path.splitext(filename)[0])
            # Save the graph using Python's pickle module
            with open(local_output_path, 'wb') as f:
                pickle.dump(graph, f)
            print(f"finished working on file: {filename}")
        else:
            print(f"Skipping file: {base_filename}" )


if __name__ == '__main__':
    start = time.time()
    run_percentiles()
    # save_netx_graphs(EMBEDDINGS_FOLDER)
    end = time.time()
    print(f"Runtime: {(end - start)/60}")
    
