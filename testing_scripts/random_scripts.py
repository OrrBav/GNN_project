import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import os
import pickle
import numpy as np

def create_graph_data(G, label, graph_name):
    # # Create a NetworkX graph
    # G = nx.from_pandas_adjacency(pd.DataFrame(adjacency_matrix))
    
    # # Add node features (embeddings) to the graph
    # for i, row in df.iterrows():
    #     G.nodes[i]['x'] = row.iloc[1:].values  # Assuming embeddings start from the second column
    
    # Convert node attributes (embeddings) from numpy arrays to PyTorch tensors
    for node in G.nodes(data=True):
        if isinstance(node[1]['embedding'], np.ndarray):
            G.nodes[node[0]]['embedding'] = torch.tensor(node[1]['embedding'], dtype=torch.float32)
    
    # Convert NetworkX graph to PyTorch Geometric Data
    data = from_networkx(G)
    
    # Assign the node embeddings to data.x
    embeddings = [G.nodes[n]['embedding'] for n in G.nodes()]
    data.x = torch.stack(embeddings).float()
    
    # Add graph label
    data.y = torch.tensor([label], dtype=torch.long)

    data.name = graph_name
    
    return data

class TCRDataset(torch.utils.data.Dataset):
    def __init__(self, graph_data_list):
        self.graph_data_list = graph_data_list

    def len(self):
        return len(self.graph_data_list)

    def get(self, idx):
        return self.graph_data_list[idx]
    

def create_pkl():
    ## SHOULD be similar to create_pt, but in pickle format.
    ## load graphs
    graph_data_list = []
    directory = '/home/dsi/orrbavly/GNN_project/data/embedding_graphs_90th_perc'
    invalid_files = ["fp", "nd", 'nh']
    for filename in os.listdir(directory):
            if all(invalid not in filename for invalid in invalid_files):
                pass


def creat_pt():
    ## load graphs
    graph_data_list = []
    directory = '/home/dsi/orrbavly/GNN_project/data/embedding_graphs'
    invalid_files = ["fp", "nd", 'nh']
    for filename in os.listdir(directory):
        if all(invalid not in filename for invalid in invalid_files):
            print(f"Working on file: {filename}")
            file_path = os.path.join(directory, filename)
            # Load the graph back into a NetworkX variable
            with open(file_path, 'rb') as f:
                G_loaded = pickle.load(f)
            label = 1 if "OC" in filename else 0
            graph_data_list.append(create_graph_data(G_loaded, label, filename.rstrip(".csv")))

    torch.save(graph_data_list, "/home/dsi/orrbavly/GNN_project/data/torch_data/83_graph_data_list.pt")


def load_dgl():
    # input should be list of netwrokx files

    import pickle
    import dgl
    # Load the NetworkX graphs
    with open('graph_data_list_networkx.pkl', 'rb') as f:
        graph_data_list = pickle.load(f)

    # Convert NetworkX graphs to DGL format
    dgl_graphs = []
    for G, label, name in graph_data_list:
        dgl_graph = dgl.from_networkx(G)
        dgl_graph.ndata['attr'] = torch.stack([G.nodes[n]['embedding'] for n in G.nodes()])
        dgl_graph.y = torch.tensor([label])
        dgl_graphs.append(dgl_graph)


def load_torch():
    # input should be list of netwrokx files
    import pickle
    from torch_geometric.utils import from_networkx

    # Load the NetworkX graphs
    with open('graph_data_list_networkx.pkl', 'rb') as f:
        graph_data_list = pickle.load(f)

    # Convert NetworkX graphs to PyTorch Geometric format
    pyg_graphs = []
    for G, label, name in graph_data_list:
        pyg_graph = from_networkx(G)
        pyg_graph.x = torch.stack([G.nodes[n]['embedding'] for n in G.nodes()])
        pyg_graph.y = torch.tensor([label])
        pyg_graphs.append(pyg_graph)


if __name__ == '__main__':
    data_path = '/home/dsi/orrbavly/GNN_project/data/torch_data/83_graph_data_list.pt'
    graph_data_list = torch.load(data_path)

    # print(f"Loaded {len(graph_data_list)} graphs")
    # # Assuming you have the list of graphs loaded in `graph_data_list`
    # for idx, data in enumerate(graph_data_list):
    #     print(f"Graph {idx}:")
        
    #     # Check number of nodes
    #     if data.x is None or data.x.size(0) == 0:
    #         print(f"  - Warning: Graph {idx} has no node features.")
    #     else:
    #         print(f"  - Number of nodes: {data.x.size(0)}")
        
    #     # Check number of edges
    #     if data.edge_index is None or data.edge_index.size(1) == 0:
    #         print(f"  - Warning: Graph {idx} has no edges.")
    #     else:
    #         print(f"  - Number of edges: {data.edge_index.size(1)}")
        
    #     # Check node features
    #     if data.x is not None:
    #         print(f"  - Node features shape: {data.x.shape}")
        
    #     # Check if graph has a label
    #     if hasattr(data, 'y'):
    #         print(f"  - Label: {data.y}")
    #     else:
    #         print(f"  - Warning: Graph {idx} has no label.")

    # print("Graph inspection complete.")