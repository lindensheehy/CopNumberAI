import json
import glob
import os

# --- Configuration ---
DATASET_DIR = "graph_dataset"

def format_custom_json(graphs, filepath):
    """Writes the JSON with indented keys, but keeps edge_index as a flat string."""
    with open(filepath, 'w') as f:
        f.write("[\n")
        
        for i, g in enumerate(graphs):
            f.write("    {\n")
            f.write(f'        "num_nodes": {g["num_nodes"]},\n')
            f.write(f'        "topology": "{g["topology"]}",\n')
            
            # Dump the massive list compactly (no spaces)
            edge_str = json.dumps(g["edge_index"], separators=(',', ':'))
            f.write(f'        "edge_index": {edge_str},\n')
            
            # Handle the cop_number (could be int or string "2+")
            cop_val = json.dumps(g.get("cop_number", "2+"))
            f.write(f'        "cop_number": {cop_val}\n')
            
            # Handle the comma for the next object
            if i < len(graphs) - 1:
                f.write("    },\n")
            else:
                f.write("    }\n")
                
        f.write("]\n")

def format_json_files():
    json_files = glob.glob(os.path.join(DATASET_DIR, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in '{DATASET_DIR}/'.")
        return

    total_files = len(json_files)

    for file_idx, filepath in enumerate(json_files, 1):
        print(f"\r  Formatting file {file_idx}/{total_files}: {os.path.basename(filepath)}...", end="", flush=True)
        
        # Read the raw data
        with open(filepath, 'r') as f:
            graphs = json.load(f)
            
        # Write it back using our custom flat-list formatter
        format_custom_json(graphs, filepath)

    print("\nSUCCESS: All JSON files have been smartly formatted!")

if __name__ == "__main__":
    format_json_files()