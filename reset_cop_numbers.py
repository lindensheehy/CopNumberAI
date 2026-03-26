import json
import glob
import os

DATASET_DIR = "graph_dataset"

def format_custom_json(graphs, filepath):
    """Writes the JSON with indented keys, keeping edge_index flat."""
    with open(filepath, 'w') as f:
        f.write("[\n")
        for i, g in enumerate(graphs):
            f.write("    {\n")
            f.write(f'        "num_nodes": {g["num_nodes"]},\n')
            f.write(f'        "topology": "{g["topology"]}",\n')
            
            edge_str = json.dumps(g["edge_index"], separators=(',', ':'))
            f.write(f'        "edge_index": {edge_str},\n')
            
            # Write the scrubbed cop_number
            f.write(f'        "cop_number": {g["cop_number"]}\n')
            
            if i < len(graphs) - 1:
                f.write("    },\n")
            else:
                f.write("    }\n")
        f.write("]\n")

def reset_dataset():
    json_files = glob.glob(os.path.join(DATASET_DIR, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {DATASET_DIR}/.")
        return

    for file_idx, filepath in enumerate(json_files, 1):
        filename = os.path.basename(filepath)
        print(f"\rScrubbing File {file_idx}/{len(json_files)}: {filename}...", end="", flush=True)
        
        with open(filepath, 'r') as f:
            graphs = json.load(f)
            
        # Hard reset every graph to -1
        for graph in graphs:
            graph["cop_number"] = -1
            
        format_custom_json(graphs, filepath)

    print("\n\nSUCCESS: All graphs have been scrubbed and reset to cop_number = -1.")

if __name__ == "__main__":
    reset_dataset()