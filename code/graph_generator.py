import networkx as nx
import json
import os
import random

# Configuration
OUTPUT_DIR = "graph_dataset"
NODE_BRACKETS = {
    10: 100,
    20: 900,
    40: 1500,
    50: 2500,
    100: 5000
}
TOPOLOGIES = ["erdos_renyi", "barabasi_albert", "watts_strogatz", "geometric"]

def generate_connected_graph(n, topology):
    """Generates a strictly connected graph with randomized parameters."""
    while True:
        if topology == "erdos_renyi":
            # Vary density: p between 0.1 (sparse) and 0.5 (dense)
            p = random.uniform(0.1, 0.5)
            G = nx.erdos_renyi_graph(n, p)
            
        elif topology == "barabasi_albert":
            # Vary hub connections: m between 1 and 4
            m = random.randint(1, min(4, n - 1))
            G = nx.barabasi_albert_graph(n, m)
            
        elif topology == "watts_strogatz":
            # Vary initial neighbors (k) and rewiring probability (p)
            k = random.randint(2, min(6, n - 1))
            p = random.uniform(0.1, 0.4)
            G = nx.connected_watts_strogatz_graph(n, k, p, tries=100) # nx has a built-in connected variant for WS
            
        elif topology == "geometric":
            # Vary radius based on theoretical connectivity threshold
            import math
            threshold = math.sqrt(math.log(n) / (math.pi * n))
            radius = random.uniform(threshold * 1.1, threshold * 2.5)
            G = nx.random_geometric_graph(n, radius)

        # The Discard & Restart Check
        if nx.is_connected(G):
            return G

def run_generator():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_generated = 0

    for n, total_graphs in NODE_BRACKETS.items():
        graphs_per_topology = total_graphs // len(TOPOLOGIES)
        
        for topology in TOPOLOGIES:
            print(f"Generating {graphs_per_topology} {topology} graphs (N={n})...")
            dataset = []
            
            for _ in range(graphs_per_topology):
                G = generate_connected_graph(n, topology)
                
                # Extract Edge Index (List of [u, v] pairs)
                # We save both directions [u,v] and [v,u] as PyTorch Geometric undirected graphs require it
                edge_index = list(G.edges())
                directed_edge_index = edge_index + [(v, u) for u, v in edge_index]
                
                graph_data = {
                    "num_nodes": n,
                    "topology": topology,
                    "edge_index": directed_edge_index
                }
                dataset.append(graph_data)
                total_generated += 1
                
            # Save to its specific grouped JSON file
            filename = f"graphs_n{n}_{topology}.json"
            filepath = os.path.join(OUTPUT_DIR, filename)
            with open(filepath, 'w') as f:
                json.dump(dataset, f)
                
            print(f"Saved {filepath}")

    print(f"\nSuccess! Generated exactly {total_generated} connected graphs across 20 files.")

if __name__ == "__main__":
    run_generator()