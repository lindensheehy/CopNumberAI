import json
import os
import subprocess
import glob
import re

# --- Configuration ---
DATASET_DIR = "graph_dataset"
TEMP_FILE = "temp_graph.txt"
EXEC_PATH = "./k_cops.exe" # Change to "./solver" if on Linux/Mac
MAX_COPS_TO_CHECK = 2  

def write_temp_matrix(num_nodes, edge_index, filename):
    matrix = [['0' for _ in range(num_nodes)] for _ in range(num_nodes)]
    for u, v in edge_index:
        matrix[u][v] = '1'
    with open(filename, 'w') as f:
        for row in matrix:
            f.write("".join(row) + "\n")
        f.write("-\n")

def check_cop_win(graph_file, k):
    try:
        result = subprocess.run(
            [EXEC_PATH, graph_file, str(k)],
            capture_output=True,
            text=True
        )
        if "RESULT: WIN" in result.stdout:
            return True
        if result.returncode == 1:
            return True
        return False
    except FileNotFoundError:
        print(f"\nCRITICAL ERROR: Executable not found at '{EXEC_PATH}'.")
        exit(1)

def format_custom_json(graphs, filepath):
    with open(filepath, 'w') as f:
        f.write("[\n")
        for i, g in enumerate(graphs):
            f.write("    {\n")
            f.write(f'        "num_nodes": {g["num_nodes"]},\n')
            f.write(f'        "topology": "{g["topology"]}",\n')
            
            edge_str = json.dumps(g["edge_index"], separators=(',', ':'))
            f.write(f'        "edge_index": {edge_str},\n')
            
            cop_val = json.dumps(g.get("cop_number", -1))
            f.write(f'        "cop_number": {cop_val}\n')
            
            if i < len(graphs) - 1:
                f.write("    },\n")
            else:
                f.write("    }\n")
        f.write("]\n")

def get_node_count(filepath):
    """Extracts the integer after 'n' in the filename for numerical sorting."""
    filename = os.path.basename(filepath)
    match = re.search(r'n(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0 # Fallback if naming convention breaks

def process_dataset():
    json_files = glob.glob(os.path.join(DATASET_DIR, "*.json"))
    
    if not json_files:
        print(f"No JSON files found.")
        return

    # --- THE SMARTER SORT ---
    # Sorts first by the numerical node count, then alphabetically by the topology name
    json_files = sorted(json_files, key=lambda x: (get_node_count(x), x))

    for file_idx, filepath in enumerate(json_files, 1):
        filename = os.path.basename(filepath)
        print(f"\n--- Processing File {file_idx}/{len(json_files)}: {filename} ---")
        
        with open(filepath, 'r') as f:
            graphs = json.load(f)
            
        total_graphs = len(graphs)
        newly_labeled = 0
        skipped = 0
        
        for i, graph in enumerate(graphs):
            if graph.get("cop_number", -1) != -1:
                skipped += 1
                continue
                
            print(f"\r  Labeling graph {i + 1}/{total_graphs} (Skipped {skipped})...", end="", flush=True)
            
            write_temp_matrix(graph["num_nodes"], graph["edge_index"], TEMP_FILE)
            
            cop_number = "2+" 
            
            for k in range(1, MAX_COPS_TO_CHECK + 1):
                if check_cop_win(TEMP_FILE, k):
                    cop_number = k
                    break
            
            graph["cop_number"] = cop_number
            newly_labeled += 1

            if newly_labeled % 100 == 0:
                format_custom_json(graphs, filepath)
                print(f"\r  [Checkpoint Saved at {i + 1}/{total_graphs}]...", end="", flush=True)

        if newly_labeled > 0:
            format_custom_json(graphs, filepath)
            print(f"\n  Finished file. Newly labeled: {newly_labeled}, Skipped: {skipped}.")
        else:
            print(f"\n  File fully computed already. Skipped all {skipped} graphs.")
            
    if os.path.exists(TEMP_FILE):
        os.remove(TEMP_FILE)
        
    print("\nSUCCESS: Dataset pipeline finished!")

if __name__ == "__main__":
    process_dataset()