import torch
import json
import glob
import os
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import degree, to_undirected

DATASET_DIR = "graph_dataset"
OUTPUT_FILE = "compiled_cops_and_robbers_v3.pt"

def compile_dataset_v3():
    print(f"Scanning '{DATASET_DIR}' for JSON graphs...")
    json_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.json")))
    
    if not json_files:
        print("No JSON files found. Exiting.")
        return

    data_list = []
    label_map = {1: 0, 2: 1, "2+": 2}
    skipped = 0

    for file_idx, filepath in enumerate(json_files, 1):
        print(f"\rProcessing File {file_idx}/{len(json_files)}...", end="", flush=True)
        with open(filepath, 'r') as f:
            graphs = json.load(f)
            
        for g_dict in graphs:
            raw_label = g_dict.get("cop_number", -1)
            if raw_label not in label_map:
                skipped += 1
                continue 
                
            y = torch.tensor([label_map[raw_label]], dtype=torch.long)
            
            edges = g_dict["edge_index"]
            if len(edges) > 0:
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                edge_index = to_undirected(edge_index)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                
            num_nodes = g_dict["num_nodes"]
            
            if edge_index.numel() > 0:
                # Feature 1: Degree
                deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)
                
                # Setup NetworkX graph for advanced features
                G = nx.Graph()
                G.add_nodes_from(range(num_nodes))
                G.add_edges_from(edges)
                
                # Feature 2: Clustering
                clust_dict = nx.clustering(G)
                clust = torch.tensor([clust_dict[i] for i in range(num_nodes)], dtype=torch.float)
                
                # Feature 3: Eigenvector Centrality
                try:
                    eig_dict = nx.eigenvector_centrality(G, max_iter=500)
                except nx.PowerIterationFailedConvergence:
                    # Fallback if the graph is too fragmented to converge
                    eig_dict = {i: 0.0 for i in range(num_nodes)}
                eig = torch.tensor([eig_dict[i] for i in range(num_nodes)], dtype=torch.float)
                
                # Stack all 3 features
                x = torch.stack([deg, clust, eig], dim=1)
            else:
                x = torch.zeros((num_nodes, 3), dtype=torch.float)
            
            data_list.append(Data(x=x, edge_index=edge_index, y=y))

    print(f"\n\nFinished processing! Skipped {skipped} uncomputed graphs.")
    print(f"Total valid graphs compiled: {len(data_list):,}")
    print(f"Saving to binary file '{OUTPUT_FILE}'...")
    torch.save(data_list, OUTPUT_FILE)
    print("SUCCESS: V3 Dataset compiled!")

if __name__ == "__main__":
    compile_dataset_v3()