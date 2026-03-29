import json
import networkx as nx
import matplotlib.pyplot as plt
import sys

class JSONGraphViewer:
    def __init__(self, filepath):
        self.filepath = filepath
        self.graphs = self.load_data()
        self.current_idx = 0
        
        # Setup Matplotlib figure and axis
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.canvas.manager.set_window_title(f"Graph Viewer - {filepath}")
        
        # Hook up the keyboard event listener
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)
        
        # Draw the initial graph
        self.draw_graph()
        plt.show()

    def load_data(self):
        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
            if not data:
                print("Error: JSON file is empty.")
                sys.exit(1)
            return data
        except FileNotFoundError:
            print(f"Error: Could not find '{self.filepath}'.")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"Error: '{self.filepath}' is not a valid JSON file.")
            sys.exit(1)

    def draw_graph(self):
        self.ax.clear()
        
        # Extract current graph data
        g_data = self.graphs[self.current_idx]
        num_nodes = g_data["num_nodes"]
        edge_index = g_data["edge_index"]
        topology = g_data.get("topology", "Unknown")
        cop_number = g_data.get("cop_number", "Uncomputed")

        # Build NetworkX Graph
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes)) # Ensure unconnected nodes still render
        G.add_edges_from(edge_index)

        # Generate a consistent layout 
        # (Seed ensures it doesn't jitter randomly if you swap back and forth)
        pos = nx.spring_layout(G, seed=42)

        # Draw using your styling preferences
        nx.draw(
            G, 
            pos, 
            ax=self.ax,
            with_labels=True, 
            node_color='skyblue', 
            node_size=300, 
            edge_color='gray', 
            font_size=8, 
            font_weight='bold'
        )

        # Update the Title with useful dataset metrics
        title = (
            f"Graph {self.current_idx + 1} / {len(self.graphs)}\n"
            f"Topology: {topology} | Nodes: {num_nodes} | Cop Number: {cop_number}"
        )
        self.ax.set_title(title, fontweight='bold', pad=15)
        
        # Add a little helper text at the bottom
        self.ax.text(
            0.5, -0.05, 
            "Use \u2190 and \u2192 arrow keys to navigate", 
            ha='center', va='center', transform=self.ax.transAxes, 
            fontsize=10, color='dimgray'
        )

        self.fig.canvas.draw()

    def on_press(self, event):
        """Handles key press events for navigation."""
        if event.key == 'right':
            # Move forward, wrap around to 0 if at the end
            self.current_idx = (self.current_idx + 1) % len(self.graphs)
            self.draw_graph()
        elif event.key == 'left':
            # Move backward, wrap around to the end if at 0
            self.current_idx = (self.current_idx - 1) % len(self.graphs)
            self.draw_graph()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python viewer.py <path_to_json_file>")
        sys.exit(1)

    json_filepath = sys.argv[1]
    viewer = JSONGraphViewer(json_filepath)