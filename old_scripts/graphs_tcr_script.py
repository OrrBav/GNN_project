import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from itertools import combinations
import tqdm
from scipy.stats import entropy
from sklearn.manifold import SpectralEmbedding
import time

# function to group sequences by their lengths
def group_sequences_by_length(sequences):
    result = {}
    for seq in sequences:
        key = len(seq)
        if key not in result:
            result[key] = set()
        result[key].add(seq)
    # Convert the sets to tuples for immutable groups
    return {k: tuple(v) for k, v in result.items()}

def generate_hashed_substrings(sequence):
    # Generate all (length-1) substrings of the current sequence and hash them
    for substring in combinations(sequence, len(sequence) - 1):
        yield hash(substring), sequence

def edit_dist_one(tcr_sequences):
     # tcr_sequences is a list.
    # Filter out invalid sequences (non-string or length <= 1)
    valid_sequences = filter(lambda s: isinstance(s, str) and len(s) > 1, tcr_sequences)
    # Group sequences by their lengths
    sequences = group_sequences_by_length(valid_sequences)
    write_edge_list_bar = tqdm.tqdm(total=sum(len(s) * l for l, s in sequences.items()), desc="Writing pairs")
    edge_list = []
    for current_length in sorted(sequences.keys(), reverse=True):
        sequences_of_current_length = tuple(map(generate_hashed_substrings, sequences[current_length]))
        sequences_of_following_length = []
        if current_length - 1 in sequences.keys():
            sequences_of_following_length = tuple(map(lambda s: hash(tuple(s)), sequences[current_length - 1]))

        for i in range(current_length - 1, -1, -1):
            # Create a dictionary to store sequences grouped by their hashed subsequences
            sequences_by_hashed_substrings = defaultdict(list)

            for hashed_subsequence, sequence in map(next, sequences_of_current_length):
                sequences_by_hashed_substrings[hashed_subsequence].append(sequence)

            # If sequences of following length exist, add them to the dictionary as indel sequences
            if current_length - 1 in sequences.keys():
                for index, hashed_sequence in enumerate(sequences_of_following_length):
                    if hashed_sequence in sequences_by_hashed_substrings:
                        sequences_by_hashed_substrings[hashed_sequence].append(sequences[current_length - 1][index])

            # Filter out groups with only one sequence
            sequences_by_hashed_substrings = [v for k, v in sequences_by_hashed_substrings.items() if len(v) > 1]

            for group in sequences_by_hashed_substrings:  # Sort sequences and generate edges
                if current_length - 1 in sequences.keys() and len(group[-1]) == current_length - 1:
                    group[:-1] = sorted(group[:-1])
                    edge_list.extend(combinations(group[:-1], 2))
                    edge_list.extend([(s, group[-1]) for s in group[:-1] if i == 0 or (i > 0 and s[i] != s[i - 1])])
                else:
                    group = sorted(group)
                    edge_list.extend(combinations(group, 2))

            write_edge_list_bar.update(len(sequences[current_length]))
        sequences.pop(current_length)
    write_edge_list_bar.close()
    # Convert the edge list to a list of tuples
    edge_list = list(edge_list)
    edge_df = pd.DataFrame(edge_list, columns=["Seq1", "Seq2"])
    print(f"Edge list - Done.")
    return edge_df

def create_edges(og_df, threshold, data_type):
    non_zero_counts = og_df.astype(bool).sum()
    filtered_columns = non_zero_counts[non_zero_counts >= threshold].index
    filtered_df = og_df[filtered_columns]
    # first column i "Unnamed:0", should be removed
    columns = filtered_df.columns[1:].tolist()
    # columns - TCR sequences serving as nodes for the graph
    # Create a new DataFrame with these columns as data under the column name "sequences"
    sequences_df = pd.DataFrame(columns, columns=['sequences'])
    # Save this DataFrame to a CSV file
    sequences_df.to_csv(f"./outputs/node_lists/node_list_{threshold}_{data_type}.csv", index=False)
    
    edge_list = edit_dist_one(columns)
    ## TODO: do i want to save edge list or node list?
    # edge_list.to_csv(f"./outputs/edge_lists/edge_list_{threshold}_{data_type}")
    return edge_list

def create_graph(edge_list):
    # edge list should be pd df, with 2 columns: Seq1, Seq2
    G = nx.Graph()
    for _, row in edge_list.iterrows():
        source = row['Seq1']
        target = row['Seq2']
        G.add_edge(source, target)
    return G

# Plots
def choose_color(graph_name):
    if graph_name.startswith('H'):
        return 'blue'
    elif graph_name.startswith('OC'):
        return 'red'
    else:
        return 'grey'

def plot_spectral_embedding(embedding, color, graph_name):
    # Plot the embedded graph
    plt.figure(figsize=(12, 10))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap=plt.cm.jet)
    plt.title(f'Spectral Embedding for {graph_name}')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plot_filename = f"./outputs/{graph_name}_spectral2D.png"
    plt.savefig(plot_filename)
    plt.close()
    return plot_filename

def plot_spectral_comb(embeddings, colors, threshold):
    plt.figure(figsize=(12, 10))
    for embedding, color in zip(embeddings, colors):
        plt.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap=plt.cm.jet, label=f'Embedding with color {color}')
    plt.title(f'Spectral Embedding for {threshold} ({colors[0]}: H, {colors[1]}: OC)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend() 
    plot_filename = f"./outputs/{threshold}_spectral2D_combined.png"
    plt.savefig(plot_filename)
    plt.close()
    return plot_filename

def plot_spectral_embedding_3d(embedding, color, graph_name):
    # Plot the embedded graph in 3D
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=color, cmap=plt.cm.jet)
    ax.title(f'Spectral Embedding for {graph_name}')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    plot_filename = f"./outputs/{graph_name}_spectral3D.png"
    plt.savefig(plot_filename)
    plt.close()
    return plot_filename


def draw_both_graphs(G1, G2, threshold):
    plt.figure(figsize=(12, 8))
    nx.draw(G1, with_labels=False, node_size=10, node_color='blue', alpha=0.6)
    nx.draw(G2, with_labels=False, node_size=10, node_color='red', alpha=0.6)
    plot_filename = f"./outputs/{threshold}_default_combined.png"
    plt.title(f"Visualization of Graphs of threshold {threshold}, using default drawing style")
    plt.savefig(plot_filename)
    plt.close()
    return plot_filename

def draw_plots(G, graph_name, color, threshold):
    # Default plot
    plt.figure(figsize=(12, 8))
    # Check if the graph name starts with 'H'
    nx.draw(G, with_labels=False, node_size=10, node_color=color)
    plt.title(f"Visualization of Graphs of threshold {threshold}, using default drawing style ({color})")

    plot_filename_default = f"./outputs/{graph_name}_default.png"
    plt.savefig(plot_filename_default)
    plt.close()

    # plt.figure(figsize=(18, 16))  # Increase figure size
    # # Define node size based on degree
    # degrees = dict(G.degree)
    # node_size = [v * 10 for v in degrees.values()]  # Adjust multiplier as needed
    # # Define edge width and color based on additional attributes if needed
    # edge_color = [0.5 if 'weight' in data else 0 for _, _, data in G.edges(data=True)]
    # edge_alpha = [0.1 if 'weight' in data else 0.5 for _, _, data in G.edges(data=True)]
    # # Spring layout
    # plt.subplot(2, 2, 1)
    # pos = nx.spring_layout(G)
    # nx.draw(G, pos, with_labels=False, node_size=node_size, edge_color=edge_color, alpha=edge_alpha)
    # plt.title("Spring Layout")
    # # Circular layout
    # plt.subplot(2, 2, 2)
    # pos = nx.circular_layout(G)
    # nx.draw(G, pos, with_labels=False, node_size=node_size, edge_color=edge_color, alpha=edge_alpha)
    # plt.title("Circular Layout")
    # # Kamada-Kawai layout
    # plt.subplot(2, 2, 3)
    # pos = nx.kamada_kawai_layout(G)
    # nx.draw(G, pos, with_labels=False, node_size=node_size, edge_color=edge_color, alpha=edge_alpha)
    # plt.title("Kamada-Kawai Layout")
    # # Set the overall title for all the subplots
    # plt.suptitle(f'Comparison of Graph Layouts for {graph_name}', fontsize=16)  # Add an overarching title for the figure
    # plt.tight_layout(pad=3.0)  # Adjust layout to make room for the title and ensure subplots do not overlap
    # plot_filename_layouts = f"./outputs/{graph_name}_layouts.png"
    # plt.savefig(plot_filename_layouts)
    # plt.close()
    plot_filename_layouts = None
    return plot_filename_default, plot_filename_layouts

def analyze_graph(G, threshold, graph_name):
    print(f"Starting analysis for threshold {threshold}")
    if nx.is_connected(G):
        diameter = nx.diameter(G)
    else:
        diameter = None
    # Calculate degree centrality for graph
    degree_centrality = nx.degree_centrality(G)
    sorted_degree_centrality = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
    top_10_degree_centrality = sorted_degree_centrality[:10]
    top_10_degree_centrality_str = ', '.join([f'"{node}":"{centrality:.3f}"' for node, centrality in top_10_degree_centrality])
    # Calculate betweenness centrality for graph
    betweenness_centrality = nx.betweenness_centrality(G)
    sorted_betweenness_centrality = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)
    top_10_betweenness_centrality = sorted_betweenness_centrality[:10]
    top_10_betweenness_centrality_str = ', '.join([f'"{node}":"{betweenness:.3f}"' for node, betweenness in top_10_betweenness_centrality])
    # Comunnities
    communities = nx.algorithms.community.greedy_modularity_communities(G)
    # Calculate the sizes of each community
    community_sizes = [(index + 1, len(community)) for index, community in enumerate(communities)]
    total_comm_size = sum(size for _, size in community_sizes)
    # Shortest paths
    shortest_paths = dict(nx.all_pairs_shortest_path_length(G))
    # Calculate average shortest path length
    total_shortest_paths = sum(sum(length for length in lengths.values()) for lengths in shortest_paths.values())
    num_pairs = sum(len(lengths) for lengths in shortest_paths.values())
    avg_shortest_path_length = total_shortest_paths / num_pairs
    # Identify and analyze subgraphs
    components = list(nx.connected_components(G))
    subgraphs = [G.subgraph(c).copy() for c in components]
    '''
    Assortativity measures the similarity of connections in the graph with respect to the node degree. It can tell you whether high-degree nodes tend
    to connect with other high-degree nodes more than would be expected by chance (assortative mixing) or the opposite (disassortative).
    '''
    assortativity = nx.degree_assortativity_coefficient(G)
    # Shannon entropy
    # Extract the degrees of all nodes
    degrees = [degree for _, degree in G.degree()]
    # Calculate the frequency of each degree value
    degree_counts = np.bincount(degrees)
    degree_probabilities = degree_counts / np.sum(degree_counts)  # Normalizing the counts to probabilities
    # Calculate Shannon entropy of the degree distribution
    diversity = entropy(degree_probabilities)
    # Calculate maximum entropy
    # Max entropy occurs when each degree probability is equal, i.e., uniform distribution among all unique degrees
    num_unique_degrees = len(degree_counts[degree_counts > 0])  # Counting only degrees that appear
    if num_unique_degrees > 0:
        max_entropy_probabilities = np.ones(num_unique_degrees) / num_unique_degrees
        max_diversity = entropy(max_entropy_probabilities)
    else:
        max_diversity = 0  # If no degrees, entropy is zero
    print(f"Starting with plots...")
    # plots
    color = choose_color(graph_name)
    plot_filename_default, plot_filename_layouts = draw_plots(G, graph_name, color, threshold)
    # spectral embedding
    print(f"Spectral Embedding...")
    try:
        spectral_2 = SpectralEmbedding(n_components=2, eigen_solver='arpack')
        adjacency_matrix_2 = nx.to_numpy_array(G)
        embedding_2 = spectral_2.fit_transform(adjacency_matrix_2)
        plot_filename_spectral2d = plot_spectral_embedding(embedding_2, color, graph_name)
    except Exception as e:
        embedding_2 = None
        error_message = str(e)
        print(f"Error during spectral embedding: {error_message}")
        plot_filename_spectral2d = f"./outputs/{graph_name}_spectral2D_error.txt"
        with open(plot_filename_spectral2d, 'w') as file:
            file.write(error_message)
       
    # spectral_3 = SpectralEmbedding(n_components=3)
    # adjacency_matrix_3 = nx.to_numpy_array(G)
    # embedding_3 = spectral_3.fit_transform(adjacency_matrix_3)
    # plot_filename_spectral3d = plot_spectral_embedding_3d(embedding_3, color, graph_name)
    plot_filename_spectral3d = None
    result = {
        'name': graph_name,
        'Threshold':threshold,
        'Number of Nodes': G.number_of_nodes(),
        'Number of Edges': G.number_of_edges(),
        'Average Node Degree': np.mean([d for _, d in G.degree]),
        'Graph density':nx.density(G),
        'Average clustering coefficient': nx.average_clustering(G),
        'Diameter': diameter,
        'Top 10 degree centrality nodes': top_10_degree_centrality_str,
        'Top 10 betweenness centrality nodes':top_10_betweenness_centrality_str,
        'Average Community size':total_comm_size / len(community_sizes),
        "Number of connected Components in graph": nx.number_connected_components(G),
        'Average shortest path length': avg_shortest_path_length,
        # add plot for path length distribution
        'Number of subgraphs':len(subgraphs),
        'Assortativity': assortativity,
        'Shannon entropy of the degree distribution': diversity,
        'Max Shannon entropy rechable': max_diversity,
        # add the different drawings
        'Default Graph plot': plot_filename_default,
        'Lauouts Graphs plots': plot_filename_layouts,
        # add spectral embeddings
        'Spectral embedding 2D plot': plot_filename_spectral2d,
        'Spectral embedding 3D plot': plot_filename_spectral3d,
        # combined plots - plotted in main
        'Combined Plot': None,
        'Combined Spectral 2D plot': None
    }
    print(f"finished working on {graph_name}.")
    return result, embedding_2

def handle_empty_graph(graph_name, threshold):
    """Return a dictionary with None values for an empty graph except for edge count message."""
    return {
        'name': graph_name,
        'Threshold': threshold,
        'Number of Nodes': 0,
        'Number of Edges': "No Edges Calculated",
        'Average Node Degree': None,
        'Graph density': None,
        'Average clustering coefficient': None,
        'Diameter': None,
        'Top 10 degree centrality nodes': None,
        'Top 10 betweenness centrality nodes': None,
        'Average Community size': None,
        "Number of connected Components in graph": 0,
        'Average shortest path length': None,
        'Number of subgraphs': 0,
        'Assortativity': None,
        'Shannon entropy of the degree distribution': None,
        'Default Graph plot': None,
        'Lauouts Graphs plots': None,
        'Spectral embedding 2D plot': None,
        'Spectral embedding 3D plot': None,
        'Combined Plot': None,
        'Combined Spectral 2D plot': None
    }, None
    # None: for embedding variable, as in analyze_graph function


if __name__ == "__main__":
    thresholds = [2, 3, 4, 5, 8, 10, 12, 15, 20, 25, 30, 32, 34]
    results = []

    # Define the file path and the chunk size
    file_path = "./data_all_ab.csv"
    chunk_size = 10000  # You can adjust this size depending on your memory and the file size

    # Determine the total number of rows (to setup the progress bar)
    total_rows = sum(1 for _ in open(file_path, 'r', encoding='utf-8')) - 1  # Subtract 1 for the header

    # Read the CSV in chunks with a progress bar
    iterator = pd.read_csv(file_path, chunksize=chunk_size, iterator=True)
    df = pd.concat(tqdm.tqdm(iterator, total=(total_rows // chunk_size) + 1, desc="Reading CSV"))

    # Filter rows where the first cell ends with "_OC"
    filtered_OC = df[df.iloc[:, 0].str.endswith("_OC")]
    # Filter rows where the first cell ends with "_H", with the same number of samples as OC
    filtered_H = df[df.iloc[:, 0].str.endswith("_H")]
    filtered_H = filtered_H.sample(n=len(filtered_OC), random_state=42)

    # create the H and OC graphs
    for threshold in reversed(thresholds):
        start_time = time.time()
        print(f"Starting threshold {threshold}")
        # Create edge lists
        edge_list_H = create_edges(filtered_H, threshold, "H")
        edge_list_OC = create_edges(filtered_OC, threshold, "OC")

        # Create and analyze H and OC graphs
        H_graph = create_graph(edge_list_H)
        OC_graph = create_graph(edge_list_OC)

        # Analyze H graph
        if H_graph.number_of_edges() == 0:
            result_H, embedding_H = handle_empty_graph(f"H_{threshold}", threshold)
            print("H_{threshold} is empty")
        else:
            result_H, embedding_H  = analyze_graph(G=H_graph, threshold=threshold, graph_name=f"H_{threshold}")
        print(f"Finished with graph H")
        # Analyze OC graph
        if OC_graph.number_of_edges() == 0:
            result_OC, embedding_OC = handle_empty_graph(f"OC_{threshold}", threshold)
            print("OC_{threshold} is empty")
        else:
            result_OC, embedding_OC = analyze_graph(G=OC_graph, threshold=threshold, graph_name=f"OC_{threshold}")
        print(f"Finished with graph OC")

        # Generate the combined plot for both graphs if both have edges
        if H_graph.number_of_edges() > 0 and OC_graph.number_of_edges() > 0:
            combined_plot_filename = draw_both_graphs(H_graph, OC_graph, threshold)
            result_H['Combined Plot'] = combined_plot_filename
            result_OC['Combined Plot'] = combined_plot_filename
        else:
            result_H['Combined Plot'] = "No combined plot available Due to 0 Edges."
            result_OC['Combined Plot'] = "No combined plot available Due to 0 Edges."

        # Generate the same for combined spectral embedding
        if embedding_H is not None and embedding_OC is not None:
            comb_spectral_filename = plot_spectral_comb([embedding_H, embedding_OC], ["blue", "red"], threshold)
            result_H['Combined Spectral 2D plot'] = comb_spectral_filename
            result_OC['Combined Spectral 2D plot'] = comb_spectral_filename
        else:
            result_H['Combined Spectral 2D plot'] = "No combined spectral 2d plot available"
            result_OC['Combined Spectral 2D plot'] = "No combined spectral 2d plot available"
        # Append the results
        results.append(result_H)
        results.append(result_OC)
        print(f"Successfully finished working on threshold {threshold}")
        
        end_time = time.time()  
        duration = (end_time - start_time) / 60 
        print(f"Threshold {threshold} processed in {duration:.2f} minutes")

    result_df = pd.DataFrame(results)
    csv_path = "./outputs/results_try2.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"CSV file has been successfully created at: {csv_path}")
    