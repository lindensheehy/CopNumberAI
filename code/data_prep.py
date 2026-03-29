import torch
import json
import glob
import os
from torch_geometric.data import Data
from torch_geometric.utils import degree, to_undirected

# --- Configuration ---
DATASET_DIR = "graph_dataset"
OUTPUT_FILE = "compiled_cops_and_robbers.pt"

def compile_dataset():
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
            
            # Skip uncomputed graphs
            if raw_label not in label_map:
                skipped += 1
                continue 
                
            # 1. Target Label (y)
            y = torch.tensor([label_map[raw_label]], dtype=torch.long)
            
            # 2. Edge Index (Bidirectional)
            edges = g_dict["edge_index"]
            if len(edges) > 0:
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                edge_index = to_undirected(edge_index)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                
            # 3. Node Features (x) -> Node Degree
            num_nodes = g_dict["num_nodes"]
            if edge_index.numel() > 0:
                deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)
                x = deg.view(-1, 1) 
            else:
                x = torch.zeros((num_nodes, 1), dtype=torch.float)
            
            # 4. Construct Data object and append
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

    print(f"\n\nFinished processing! Skipped {skipped} uncomputed graphs.")
    print(f"Total valid graphs compiled: {len(data_list):,}")
    
    # --- SAVE TO BINARY ---
    print(f"Saving to binary file '{OUTPUT_FILE}'...")
    torch.save(data_list, OUTPUT_FILE)
    print("SUCCESS: Dataset compiled and ready for training!")

if __name__ == "__main__":
    compile_dataset()