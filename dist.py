import json
import os
import glob
from collections import defaultdict

# --- Configuration ---
DATASET_DIR = "graph_dataset"

def analyze_distribution():
    json_files = glob.glob(os.path.join(DATASET_DIR, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in '{DATASET_DIR}/'.")
        return

    # Master counters
    total_counts = {"1": 0, "2": 0, "2+": 0, "Uncomputed (-1)": 0}
    
    # Nested dictionary for topology-specific counts
    topology_counts = defaultdict(lambda: {"1": 0, "2": 0, "2+": 0, "Uncomputed (-1)": 0})
    
    total_graphs = 0

    print("Scanning dataset... Please wait.")

    for filepath in json_files:
        try:
            with open(filepath, 'r') as f:
                graphs = json.load(f)
        except json.JSONDecodeError:
            print(f"Error reading {filepath}. Skipping.")
            continue

        for g in graphs:
            total_graphs += 1
            raw_cop = g.get("cop_number", -1)
            topo = g.get("topology", "Unknown")

            # Normalize the keys (handles integers vs strings)
            if raw_cop == 1:
                key = "1"
            elif raw_cop == 2:
                key = "2"
            elif raw_cop == "2+":
                key = "2+"
            elif raw_cop == -1:
                key = "Uncomputed (-1)"
            else:
                key = str(raw_cop)
                if key not in total_counts:
                    total_counts[key] = 0
                    topology_counts[topo][key] = 0

            # Increment global and topology-specific counters
            total_counts.setdefault(key, 0)
            total_counts[key] += 1
            
            topology_counts[topo].setdefault(key, 0)
            topology_counts[topo][key] += 1

    # --- FORMATTED OUTPUT ---
    print("\n" + "=" * 50)
    print(" 📊 DATASET LABEL DISTRIBUTION 📊")
    print("=" * 50)
    print(f"Total Graphs Scanned: {total_graphs:,}\n")

    print("--- GLOBAL TOTALS ---")
    for k in ["1", "2", "2+", "Uncomputed (-1)"]:
        v = total_counts.get(k, 0)
        pct = (v / total_graphs) * 100 if total_graphs > 0 else 0
        print(f"  Cop Number {k:<15}: {v:>7,} ({pct:>5.1f}%)")

    print("\n--- BREAKDOWN BY TOPOLOGY ---")
    for topo, counts in topology_counts.items():
        print(f"\n{topo.upper()}:")
        topo_total = sum(counts.values())
        for k in ["1", "2", "2+", "Uncomputed (-1)"]:
            v = counts.get(k, 0)
            if v > 0 or k in ["1", "2", "2+"]: # Only print non-zero uncomputed stats
                pct = (v / topo_total) * 100 if topo_total > 0 else 0
                print(f"  Cop {k:<10}: {v:>7,} ({pct:>5.1f}%)")

if __name__ == "__main__":
    analyze_distribution()