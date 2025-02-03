import os
import pickle
import networkx as nx
import pandas as pd
import numpy as np

# Function to load all NetworkX graphs from a given directory
def load_graphs_from_dir(directory_path):
    graphs = []
    
    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        if 'fp' not in filename:
            file_path = os.path.join(directory_path, filename)
            base_filename = os.path.splitext(filename)[0]
            # Load the NetworkX graph from the pickle file
            with open(file_path, 'rb') as f:
                graph = pickle.load(f)
                if isinstance(graph, nx.Graph):
                    graphs.append((graph, base_filename))
                else:
                    print(f"File {filename} does not contain a valid NetworkX graph.")
    
    return graphs   

# Function to accumulate centrality information and store results in a Pandas DataFrame
def accumulate_tcr_centrality_to_df(graphs_with_names, tcr_sequences):
    # graphs_with_names: list of tuples in format (graph, graph_name)
    results = []

    # Iterate through each graph
    # Iterate through each graph and its corresponding name from the tuple
    for graph, graph_name in graphs_with_names:
        print(f"working on graph: {graph_name}")
        # Calculate centrality and other importance measures for the graph
        print("Degree")
        degree_centrality = nx.degree_centrality(graph)
        print("betweenes")
        # Using approximation for betweenness centrality to speed up computation
        betweenness_centrality = nx.betweenness_centrality(graph, k=1000)  # Approximate with 100 random nodes
        print("closeness")
        closeness_centrality = nx.closeness_centrality(graph)
        print("eigenvector")
        # Adjusting eigenvector centrality to handle convergence issues with approximation
        try:
            # Try with initial parameters; these might converge close enough without being perfect
            eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=500, tol=1e-5)
        except nx.PowerIterationFailedConvergence:
            print(f"Eigenvector centrality failed to converge with default settings for graph: {graph_name}")
            try:
                # If the first attempt fails, retry with looser tolerance for faster convergence
                eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=300, tol=1e-2)
                print(f"Eigenvector centrality approximated for graph: {graph_name} with higher tolerance.")
            except nx.PowerIterationFailedConvergence:
                # If it still fails, fallback to an empty result or a different measure
                eigenvector_centrality = {}
                print(f"Eigenvector centrality could not be approximated for graph: {graph_name}")
        print("pagerank")
        pagerank = nx.pagerank(graph, max_iter=500, tol=1e-6)  # Adjusted max_iter and tol for better convergence
        print("clustering")
        clustering_coefficient = nx.clustering(graph)
        # print("structural")
        # # Calculating structural holes constraint
        # structural_holes_constraint = nx.constraint(graph)

        print("norm")
        # Calculate average centrality values for all nodes as the "norm"
        all_nodes = list(graph.nodes())
        
        # Collect values for each centrality measure
        norm_degree = np.mean([degree_centrality[node] for node in all_nodes])
        norm_betweenness = np.mean([betweenness_centrality[node] for node in all_nodes])
        norm_closeness = np.mean([closeness_centrality[node] for node in all_nodes])
        norm_eigenvector = np.mean([eigenvector_centrality.get(node, 0) for node in all_nodes])
        norm_pagerank = np.mean([pagerank[node] for node in all_nodes])
        norm_clustering = np.mean([clustering_coefficient[node] for node in all_nodes])
        # norm_constraint = np.mean([structural_holes_constraint.get(node, 0) for node in all_nodes])

        results.append({
            'graph_name': graph_name,
            'tcr_sequence': "mean",
            'degree_centrality': norm_degree,
            'betweenness_centrality': norm_betweenness,
            'closeness_centrality': norm_closeness,
            'eigenvector_centrality': norm_eigenvector,
            'pagerank': norm_pagerank,
            'clustering_coefficient': norm_clustering,
            # 'constraint': constraint,
        })
        # Check for each TCR sequence in the graph
        for tcr in tcr_sequences:
            print(f"working on tcr: {tcr}")
            if tcr in graph:
                # Get individual centrality metrics safely using `.get()`
                degree = degree_centrality.get(tcr, 0)
                betweenness = betweenness_centrality.get(tcr, 0)
                closeness = closeness_centrality.get(tcr, 0)
                eigenvector = eigenvector_centrality.get(tcr, 0)
                page_rank = pagerank.get(tcr, 0)
                clustering = clustering_coefficient.get(tcr, 0)
                # constraint = structural_holes_constraint.get(tcr, 0)
                
                # Add the centrality metrics to the results list
                results.append({
                    'graph_name': graph_name,
                    'tcr_sequence': tcr,
                    'degree_centrality': degree,
                    'betweenness_centrality': betweenness,
                    'closeness_centrality': closeness,
                    'eigenvector_centrality': eigenvector,
                    'pagerank': page_rank,
                    'clustering_coefficient': clustering,
                    # 'constraint': constraint,
                })
            else:
                print(f"TCR {tcr} not found in graph {graph_name}.")
                # If the TCR sequence is not present in the graph, add a row with NaN values
                results.append({
                    'graph_name': graph_name,
                    'tcr_sequence': tcr,
                    'degree_centrality': None,
                    'betweenness_centrality': None,
                    'closeness_centrality': None,
                    'eigenvector_centrality': None,
                    'pagerank': None,
                    'clustering_coefficient': None,
                    # 'constraint': None,
                })

    # Convert the results into a DataFrame
    df = pd.DataFrame(results)
    
    return df

if __name__ == "__main__":
    graphs = load_graphs_from_dir('/home/dsi/orrbavly/GNN_project/data/embedding_graphs_90th_perc_new')
    tcr_sequences = ['CAVRDSNYQLIW', 'CVVSDRGSTLGRLYF', 'CAVLDSNYQLIW', 'CASSLGETQYF']
    # Accumulate the centrality information into a DataFrame
    centrality_df = accumulate_tcr_centrality_to_df(graphs, tcr_sequences)
    # Save DataFrame to a CSV file
    centrality_df.to_csv('/home/dsi/orrbavly/GNN_project/outputs/centrality_results.csv', index=False)
