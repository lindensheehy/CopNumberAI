import torch
import json
import glob
import os
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import degree, to_undirected

# --- Configuration ---
DATASET_DIR = "graph_dataset"
OUTPUT_FILE = "compiled_cops_and_robbers_v2.pt" # <--- New V2 file

def compile_dataset_v2():
    print(f"Scanning '{DATASET_DIR}' for JSON graphs...")
    json_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.json")))
    
    if not json_files:
        print("No JSON files found. Exiting.")
        return

    data_list = []
    label_map = {1: 0, 2: 1, "2+": 2}
    skipped = 0

    for file_idx, filepath in enumerate(json_files, 1):
        print(f"\rProcessing File {file_idx}/{len(json_files)}: {os.path.basename(filepath)}...", end="", flush=True)
        
        with open(filepath, 'r') as f:
            graphs = json.load(f)
            
        for g_dict in graphs:
            raw_label = g_dict.get("cop_number", -1)
            if raw_label not in label_map:
                skipped += 1
                continue 
                
            y = torch.tensor([label_map[raw_label]], dtype=torch.long)
            
            # Edges
            edges = g_dict["edge_index"]
            if len(edges) > 0:
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                edge_index = to_undirected(edge_index)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                
            num_nodes = g_dict["num_nodes"]
            
            if edge_index.numel() > 0:
                # Feature 1: Node Degree
                deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)
                
                # Feature 2: Clustering Coefficient (using NetworkX)
                # We build a temporary NetworkX graph just to run their highly-optimized math
                G = nx.Graph()
                G.add_nodes_from(range(num_nodes))
                G.add_edges_from(edges)
                
                # Calculate clustering for all nodes, returns a dict {node_id: float}
                clustering_dict = nx.clustering(G)
                # Convert the dict into a flat list, ordered by node ID, then into a tensor
                clust = torch.tensor([clustering_dict[i] for i in range(num_nodes)], dtype=torch.float)
                
                # Stack them together into a matrix of shape [num_nodes, 2]
                x = torch.stack([deg, clust], dim=1)
                
            else:
                # Fallback for empty graphs
                x = torch.zeros((num_nodes, 2), dtype=torch.float)
            
            data_list.append(Data(x=x, edge_index=edge_index, y=y))

    print(f"\n\nFinished processing! Skipped {skipped} uncomputed graphs.")
    print(f"Total valid graphs compiled: {len(data_list):,}")
    
    print(f"Saving to binary file '{OUTPUT_FILE}'...")
    torch.save(data_list, OUTPUT_FILE)
    print("SUCCESS: V2 Dataset compiled!")

if __name__ == "__main__":
    compile_dataset_v2()