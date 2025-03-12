import os
import networkx as nx
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def read_patterns(file_path):
    """
    Read patterns from the given file path.
    Returns a list of patterns, where each pattern is a list of edges.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Get the number of patterns
    num_patterns = int(lines[0].strip())
    
    patterns = []
    current_pattern = None
    pattern_index = -1  # To track which pattern we're reading
    
    line_index = 1  # Skip the first line (number of patterns)
    while line_index < len(lines):
        line = lines[line_index].strip()
        line_index += 1
        
        # Empty line indicates end of a pattern or start of a new one
        if not line:
            continue
        
        # If we can parse line as an integer, it's the pattern index
        try:
            pattern_index = int(line)
            current_pattern = []
            patterns.append(current_pattern)
            continue
        except ValueError:
            pass
        
        # Otherwise, it's an edge description
        if current_pattern is not None:
            source, dest = map(int, line.split())
            current_pattern.append((source, dest))
    
    return patterns[:num_patterns]  # Ensure we only return the expected number of patterns

def find_critical_path(G):
    """
    Find the critical path from node 0 to node 1 using topological sorting.
    Returns the path as a list of nodes and the length of the path.
    """
    # Check if a path exists from 0 to 1
    if not nx.has_path(G, 0, 1):
        return None, 0
    
    # Topological sort
    topo_order = list(nx.topological_sort(G))
    
    # Initialize distances
    dist = {node: float('-inf') for node in G.nodes()}
    dist[0] = 0  # Start node
    
    # Initialize predecessor tracking
    pred = {node: None for node in G.nodes()}
    
    # Dynamic programming to find longest path
    for node in topo_order:
        if dist[node] != float('-inf'):  # If we've reached this node
            for neighbor in G.successors(node):
                if dist[node] + 1 > dist[neighbor]:  # Edge weight is 1
                    dist[neighbor] = dist[node] + 1
                    pred[neighbor] = node
    
    # Reconstruct path
    if dist[1] == float('-inf'):  # No path to node 1
        return None, 0
    
    path = []
    current = 1
    while current is not None:
        path.append(current)
        current = pred[current]
    
    path.reverse()  # Reverse to get path from 0 to 1
    return path, dist[1]

class DAGVisualizerApp:
    def __init__(self, root, patterns, file_path=None):
        self.root = root
        self.patterns = patterns  # list of pattern edge lists
        self.file_path = file_path
        self.current_pattern_idx = 0
        self.graphs = []  # Store NetworkX graph objects
        
        # Create graph objects for all patterns
        for pattern in patterns:
            G = nx.DiGraph()
            for source, dest in pattern:
                G.add_edge(source, dest)
            self.graphs.append(G)

        # Register window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.root.title("DAG Pattern Visualizer")
        self.root.geometry("1000x800")

        # Control frame at the top
        self.control_frame = ttk.Frame(root, padding="10")
        self.control_frame.pack(fill=tk.X)
        
        # Using pattern from file
        ttk.Label(self.control_frame, text="File:").pack(side=tk.LEFT, padx=(0,1))
        file_name = os.path.basename(file_path) if file_path else "No file selected"
        ttk.Label(self.control_frame, text=file_name).pack(side=tk.LEFT)

        # Add an extending spacer
        ttk.Label(self.control_frame, text="").pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Dropdown to select pattern
        ttk.Label(self.control_frame, text="Select Pattern:").pack(side=tk.LEFT, padx=(0,10))
        self.pattern_var = tk.StringVar()
        self.pattern_selector = ttk.Combobox(
            self.control_frame,
            textvariable=self.pattern_var,
            values=[f"Pattern {i}" for i in range(len(patterns))],
            width=15
        )
        self.pattern_selector.current(0)
        self.pattern_selector.pack(side=tk.LEFT)
        self.pattern_selector.bind("<<ComboboxSelected>>", self.on_pattern_selected)
        
        # Navigation buttons
        self.prev_button = ttk.Button(self.control_frame, text="Previous", command=self.prev_pattern)
        self.prev_button.pack(side=tk.LEFT, padx=10)
        self.next_button = ttk.Button(self.control_frame, text="Next", command=self.next_pattern)
        self.next_button.pack(side=tk.LEFT, padx=10)
        
        # Layout selector
        ttk.Label(self.control_frame, text="Layout:").pack(side=tk.LEFT, padx=(10,5))
        self.layout_var = tk.StringVar(value="dot")
        self.layout_selector = ttk.Combobox(
            self.control_frame,
            textvariable=self.layout_var,
            values=["dot", "neato", "fdp", "sfdp", "twopi", "circo", "spring"],
            width=10
        )
        self.layout_selector.pack(side=tk.LEFT)
        self.layout_selector.bind("<<ComboboxSelected>>", self.update_plot)
        
        # Layout info button
        self.layout_info_button = ttk.Button(self.control_frame, text="?", width=2, 
                                            command=self.show_layout_info)
        self.layout_info_button.pack(side=tk.LEFT, padx=2)
        
        # Index display label
        self.index_label = ttk.Label(self.control_frame, text=f"Pattern {self.current_pattern_idx}/{len(patterns)-1}")
        self.index_label.pack(side=tk.LEFT, padx=20)
        
        # Info frame
        self.info_frame = ttk.Frame(root, padding="5")
        self.info_frame.pack(fill=tk.X)
        
        # Node count
        self.node_label = ttk.Label(self.info_frame, text="Nodes: 0")
        self.node_label.pack(side=tk.LEFT, padx=(0,20))
        
        # Edge count
        self.edge_label = ttk.Label(self.info_frame, text="Edges: 0")
        self.edge_label.pack(side=tk.LEFT, padx=(0,20))
        
        # Is DAG
        self.dag_label = ttk.Label(self.info_frame, text="Is DAG: Yes")
        self.dag_label.pack(side=tk.LEFT, padx=(0,20))
        
        # Critical path
        self.path_label = ttk.Label(self.info_frame, text="Critical Path: None")
        self.path_label.pack(side=tk.LEFT, padx=(0,20))
        
        # Create matplotlib figure and canvas
        self.fig, self.ax = plt.subplots(figsize=(10,8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Display the first pattern
        self.update_plot()
        self.update_button_states()
        
    def hierarchical_layout(self, G, horizontal=False):
        """
        Custom hierarchical layout for DAGs without requiring GraphViz.
        Places nodes in layers based on their topological position.
        
        Args:
            G: The graph to lay out
            horizontal: If True, layout flows left to right instead of top to bottom
        """
        # Ensure we have a DAG
        if not nx.is_directed_acyclic_graph(G):
            return nx.spring_layout(G)  # Fallback for non-DAGs
            
        # Get topological generations (nodes grouped by their layer)
        try:
            generations = list(nx.topological_generations(G))
        except:
            # If topological_generations fails
            return nx.spring_layout(G)
            
        # Position nodes in layers
        pos = {}
        layer_count = len(generations)
        
        # Special case: if only input and output nodes
        if layer_count <= 1:
            return nx.spring_layout(G)
            
        # Otherwise place nodes in layers
        for i, layer in enumerate(generations):
            # For horizontal layout, x is the primary progression direction
            if horizontal:
                x = i / (layer_count - 1) if layer_count > 1 else 0.5
                # Place nodes vertically in each layer
                node_count = len(layer)
                for j, node in enumerate(sorted(layer)):
                    y = 1.0 - (j + 0.5) / node_count if node_count > 1 else 0.5
                    pos[node] = (x, y)
            else:
                # Traditional top-to-bottom layout
                y = 1.0 - i / (layer_count - 1) if layer_count > 1 else 0.5
                # Place nodes horizontally in each layer
                node_count = len(layer)
                for j, node in enumerate(sorted(layer)):
                    x = (j + 0.5) / node_count if node_count > 1 else 0.5
                    pos[node] = (x, y)
                
        return pos
        
    def get_node_shells(self, G):
        """
        Group nodes into shells based on their distance from the input node (0).
        Ensures every node is included in exactly one shell.
        """
        # If node 0 doesn't exist, return a default shell arrangement
        if 0 not in G.nodes():
            return [list(G.nodes())]
            
        # Track which nodes we've assigned to shells
        assigned = set()
        shells = []
        
        # First shell is just the input node
        shells.append([0])
        assigned.add(0)
        
        # Use BFS to find nodes by distance from input
        current_shell = [0]
        
        while current_shell:
            next_shell = []
            
            # Find successors of current shell nodes
            for node in current_shell:
                for successor in G.successors(node):
                    if successor not in assigned:
                        next_shell.append(successor)
                        assigned.add(successor)
            
            # If we found any new nodes, add them as a shell
            if next_shell:
                shells.append(next_shell)
                current_shell = next_shell
            else:
                current_shell = []
        
        # Add any remaining unassigned nodes as a final shell
        remaining = [n for n in G.nodes() if n not in assigned]
        if remaining:
            shells.append(remaining)
            
        # Empty shells can cause problems, so filter them out
        shells = [shell for shell in shells if shell]
        
        # If we ended up with no shells (shouldn't happen), use a fallback
        if not shells:
            shells = [list(G.nodes())]
            
        return shells

    def on_closing(self):
        plt.close(self.fig)
        self.root.destroy()

    def update_plot(self, event=None):
        self.ax.clear()
        
        G = self.graphs[self.current_pattern_idx]
        is_dag = nx.is_directed_acyclic_graph(G)
        
        # Calculate node positions based on selected layout
        layout_type = self.layout_var.get()
        
        try:
            # Use pure NetworkX layouts since we're on Windows without GraphViz
            if layout_type == "dot":
                # Simple hierarchical layout using topological generations
                pos = self.hierarchical_layout(G, horizontal=True)
                layout_status = "hierarchical (NetworkX)"
            elif layout_type == "circo":
                # Circular layout - arrange nodes in a circle
                pos = nx.circular_layout(G)
                layout_status = "circular (NetworkX)"
            elif layout_type == "twopi":
                # Shell layout - concentric circles arranged by path length from input
                try:
                    # Ensure all nodes are in the pos dictionary
                    shells = self.get_node_shells(G)
                    pos = nx.shell_layout(G, shells)
                    layout_status = "shell (NetworkX)"
                except Exception as e:
                    # If any error in shell layout, fall back to circular
                    print(f"Shell layout error: {str(e)}")
                    pos = nx.circular_layout(G)
                    layout_status = "circular (NetworkX)"
            elif layout_type == "neato":
                # Kamada-Kawai layout - force-directed
                pos = nx.kamada_kawai_layout(G)
                layout_status = "kamada-kawai (NetworkX)"
            elif layout_type == "fdp":
                # Spring layout with more iterations and stronger repulsion
                pos = nx.spring_layout(G, k=1.5, iterations=100, seed=42)
                layout_status = "spring-strong (NetworkX)"
            elif layout_type == "sfdp":
                # Spectral layout
                pos = nx.spectral_layout(G)
                layout_status = "spectral (NetworkX)"
            else:
                # Default to spring layout
                pos = nx.spring_layout(G, seed=42)
                layout_status = "spring (NetworkX)"
                
            # Ensure all nodes have positions
            for node in G.nodes():
                if node not in pos or pos[node] is None:
                    # Assign a random position for any missing nodes
                    pos[node] = (np.random.random(), np.random.random())
        except Exception as e:
            # Final fallback for any errors
            print(f"Layout error: {str(e)}")
            pos = nx.spring_layout(G, seed=42)
            layout_status = "spring fallback (NetworkX)"
            
        # Find critical path
        critical_path, path_length = find_critical_path(G)
        
        # Set node colors
        node_colors = []
        for node in G.nodes():
            if node == 0:  # Input node
                node_colors.append('green')
            elif node == 1:  # Output node
                node_colors.append('red')
            elif critical_path and node in critical_path:  # On critical path
                node_colors.append('orange')
            else:  # Other nodes
                node_colors.append('skyblue')
        
        # Set edge colors
        edge_colors = []
        for u, v in G.edges():
            if critical_path and u in critical_path and v in critical_path and critical_path.index(v) == critical_path.index(u) + 1:
                edge_colors.append('red')  # On critical path
            else:
                edge_colors.append('gray')
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                              node_color=node_colors, 
                              node_size=500, 
                              ax=self.ax)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, 
                              edge_color=edge_colors, 
                              width=1.5, 
                              arrowsize=15,
                              ax=self.ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_weight='bold', ax=self.ax)
        
        # Set title and remove axis
        self.ax.set_title(f"Pattern {self.current_pattern_idx} - Layout: {layout_status}")
        self.ax.axis('off')
        self.fig.tight_layout()
        
        # Update labels
        self.index_label.config(text=f"Pattern {self.current_pattern_idx}/{len(self.patterns)-1}")
        self.node_label.config(text=f"Nodes: {G.number_of_nodes()}")
        self.edge_label.config(text=f"Edges: {G.number_of_edges()}")
        self.dag_label.config(text=f"Is DAG: {'Yes' if is_dag else 'No'}")
        
        if critical_path:
            path_str = " → ".join(map(str, critical_path))
            self.path_label.config(text=f"Critical Path: {path_str} (Length: {path_length})")
        else:
            self.path_label.config(text="Critical Path: None")
        
        # Redraw canvas
        self.canvas.draw()
        
        # Update labels
        self.index_label.config(text=f"Pattern {self.current_pattern_idx}/{len(self.patterns)-1}")
        self.node_label.config(text=f"Nodes: {G.number_of_nodes()}")
        self.edge_label.config(text=f"Edges: {G.number_of_edges()}")
        self.dag_label.config(text=f"Is DAG: {'Yes' if is_dag else 'No'}")
        
        if critical_path:
            path_str = " → ".join(map(str, critical_path))
            self.path_label.config(text=f"Critical Path: {path_str} (Length: {path_length})")
        else:
            self.path_label.config(text="Critical Path: None")
        
        # Redraw canvas
        self.canvas.draw()
            
        # Add status message to title
        layout_status = layout_type
        if layout_type != "spring" and layout_type + " (fallback to spring)" in str(pos):
            layout_status = f"{layout_type} (fallback to spring - GraphViz not available)"
        
        # Find critical path
        critical_path, path_length = find_critical_path(G)
        
        # Set node colors
        node_colors = []
        for node in G.nodes():
            if node == 0:  # Input node
                node_colors.append('green')
            elif node == 1:  # Output node
                node_colors.append('red')
            elif critical_path and node in critical_path:  # On critical path
                node_colors.append('orange')
            else:  # Other nodes
                node_colors.append('skyblue')
        
        # Set edge colors
        edge_colors = []
        for u, v in G.edges():
            if critical_path and u in critical_path and v in critical_path and critical_path.index(v) == critical_path.index(u) + 1:
                edge_colors.append('red')  # On critical path
            else:
                edge_colors.append('gray')
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                              node_color=node_colors, 
                              node_size=500, 
                              ax=self.ax)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, 
                              edge_color=edge_colors, 
                              width=1.5, 
                              arrowsize=15,
                              ax=self.ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_weight='bold', ax=self.ax)
        
        # Add status message to title
        layout_status = layout_type
        if isinstance(pos, dict) and all(isinstance(p, list) or isinstance(p, tuple) for p in pos.values()):
            # Seems we got valid positions
            layout_status = layout_type
        else:
            layout_status = f"{layout_type} (fallback to spring)"
            
        # Find critical path
        critical_path, path_length = find_critical_path(G)
        
        # Set node colors
        node_colors = []
        for node in G.nodes():
            if node == 0:  # Input node
                node_colors.append('green')
            elif node == 1:  # Output node
                node_colors.append('red')
            elif critical_path and node in critical_path:  # On critical path
                node_colors.append('orange')
            else:  # Other nodes
                node_colors.append('skyblue')
        
        # Set edge colors
        edge_colors = []
        for u, v in G.edges():
            if critical_path and u in critical_path and v in critical_path and critical_path.index(v) == critical_path.index(u) + 1:
                edge_colors.append('red')  # On critical path
            else:
                edge_colors.append('gray')
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                              node_color=node_colors, 
                              node_size=500, 
                              ax=self.ax)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, 
                              edge_color=edge_colors, 
                              width=1.5, 
                              arrowsize=15,
                              ax=self.ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_weight='bold', ax=self.ax)
        
        # Set title and remove axis
        self.ax.set_title(f"Pattern {self.current_pattern_idx} - Layout: {layout_status}")
        self.ax.axis('off')
        self.fig.tight_layout()
        
        # Update labels
        self.index_label.config(text=f"Pattern {self.current_pattern_idx}/{len(self.patterns)-1}")
        self.node_label.config(text=f"Nodes: {G.number_of_nodes()}")
        self.edge_label.config(text=f"Edges: {G.number_of_edges()}")
        self.dag_label.config(text=f"Is DAG: {'Yes' if is_dag else 'No'}")
        
        if critical_path:
            path_str = " → ".join(map(str, critical_path))
            self.path_label.config(text=f"Critical Path: {path_str} (Length: {path_length})")
        else:
            self.path_label.config(text="Critical Path: None")
        
        # Redraw canvas
        self.canvas.draw()
    
    def update_button_states(self):
        self.prev_button.config(state=tk.NORMAL if self.current_pattern_idx > 0 else tk.DISABLED)
        self.next_button.config(state=tk.NORMAL if self.current_pattern_idx < len(self.patterns) - 1 else tk.DISABLED)
    
    def on_pattern_selected(self, event):
        idx = self.pattern_selector.current()
        if 0 <= idx < len(self.patterns):
            self.current_pattern_idx = idx
            self.update_plot()
            self.update_button_states()
    
    def next_pattern(self):
        if self.current_pattern_idx < len(self.patterns) - 1:
            self.current_pattern_idx += 1
            self.pattern_selector.current(self.current_pattern_idx)  # Update dropdown selection
            self.update_plot()
            self.update_button_states()
    
    def prev_pattern(self):
        if self.current_pattern_idx > 0:
            self.current_pattern_idx -= 1
            self.pattern_selector.current(self.current_pattern_idx)  # Update dropdown selection
            self.update_plot()
            self.update_button_states()
            
    def show_layout_info(self):
        """Display information about the different layout algorithms."""
        layout_info = (
            "Layout Algorithms (Windows-compatible version):\n\n"
            "• dot: Hierarchical layout - Custom implementation using topological generations\n"
            "• neato: Force-directed using Kamada-Kawai algorithm\n"
            "• fdp: Enhanced spring layout with stronger repulsion\n"
            "• sfdp: Spectral layout - Uses eigenvectors of graph Laplacian\n"
            "• twopi: Shell layout - Nodes arranged in concentric circles by graph distance\n"
            "• circo: Circular layout - All nodes arranged in a single circle\n"
            "• spring: Standard spring layout with moderate settings\n\n"
            "These are pure Python implementations using NetworkX that don't require GraphViz.\n"
            "For timing analysis, 'dot' is usually best as it shows the hierarchical flow."
        )
        messagebox.showinfo("Layout Information", layout_info)

def main():
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="Visualize DAG patterns from a text file")
    parser.add_argument("file_path", help="Path to the input file with DAG patterns")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of patterns to process")
    
    args = parser.parse_args()
    
    try:
        # Read patterns from the file
        print(f"Reading patterns from {args.file_path}...")
        patterns = read_patterns(args.file_path)
        
        # Limit the number of patterns if requested
        if args.limit and args.limit < len(patterns):
            print(f"Limiting to first {args.limit} patterns out of {len(patterns)}")
            patterns = patterns[:args.limit]
        else:
            print(f"Found {len(patterns)} patterns")
        
        # Create and run the GUI
        root = tk.Tk()
        app = DAGVisualizerApp(root, patterns, args.file_path)
        root.mainloop()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()