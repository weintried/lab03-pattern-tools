import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import random
from matplotlib.path import Path
import matplotlib.patches as patches

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

def read_weights(file_path):
    """
    Read node weights from the given file path.
    Returns a list of weight dictionaries, where each dictionary maps node IDs to weights.
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            return None
            
        # Get the number of patterns
        num_patterns = int(lines[0].strip())
        weights = []
        
        line_idx = 1
        pattern_idx = 0
        
        # Skip the initial blank line if present
        if line_idx < len(lines) and not lines[line_idx].strip():
            line_idx += 1
        
        while pattern_idx < num_patterns and line_idx < len(lines):
            # Read pattern index
            if not lines[line_idx].strip():
                line_idx += 1
                continue
            
            try:
                current_pattern_idx = int(lines[line_idx].strip())
                line_idx += 1
                
                # Create a new weight dictionary for this pattern
                current_weights = {}
                weights.append(current_weights)
                
                # Read 16 weights (nodes 0-15)
                for node_idx in range(16):
                    if line_idx < len(lines) and lines[line_idx].strip():
                        try:
                            weight = int(lines[line_idx].strip())
                            current_weights[node_idx] = weight
                        except ValueError:
                            pass  # Skip invalid weights
                    line_idx += 1
                
                pattern_idx += 1
                
            except ValueError:
                line_idx += 1  # Skip lines that can't be parsed as integers
        
        # Debug prints
        print(f"Read {len(weights)} weight patterns")
        if weights:
            print(f"First pattern weights: {weights[0]}")
        
        return weights
    
    except (FileNotFoundError, ValueError, IndexError) as e:
        print(f"Error reading weights file: {str(e)}")
        return None

def write_weights(file_path, all_weights, num_patterns):
    """
    Write node weights to the given file path in the required format.
    
    Args:
        file_path: Path to write the weights file
        all_weights: List of dictionaries, each mapping node IDs to weights
        num_patterns: Total number of patterns
    """
    with open(file_path, 'w') as f:
        # Write number of patterns
        f.write(f"{num_patterns}\n\n")
        
        # Write each pattern's weights
        for pattern_idx, weights in enumerate(all_weights):
            # Write pattern index
            f.write(f"{pattern_idx}\n")
            
            # Write weights for each node (assuming 16 nodes per pattern)
            for node_idx in range(16):  # Nodes 0-15
                # Use stored weight or generate a new random weight, but never default to 1
                if node_idx in weights:
                    weight = weights[node_idx]
                else:
                    # This shouldn't happen with proper weight initialization, but just in case
                    weight = random.randint(0, 15)
                    
                f.write(f"{weight}\n")
            
            # Add blank line between patterns
            f.write("\n")

def generate_random_weights(num_patterns, num_nodes=16, min_val=0, max_val=15):
    """
    Generate random weights for all nodes in all patterns.
    
    Args:
        num_patterns: Number of patterns
        num_nodes: Number of nodes per pattern
        min_val: Minimum weight value
        max_val: Maximum weight value
        
    Returns:
        List of dictionaries mapping node IDs to weights
    """
    all_weights = []
    
    for _ in range(num_patterns):
        weights = {node: random.randint(min_val, max_val) for node in range(num_nodes)}
        all_weights.append(weights)
    
    return all_weights

def find_all_critical_paths(G):
    """
    Find all critical paths from node 0 to node 1 based on accumulating node weights.
    Returns a list of all paths with the maximum weight and the total weight.
    """
    # Check if a path exists from 0 to 1
    if not nx.has_path(G, 0, 1):
        return [], 0
    
    # Topological sort
    topo_order = list(nx.topological_sort(G))
    
    # Get node weights with a default value of 1 if weight is not assigned
    weights = {node: G.nodes[node].get('weight', 1) for node in G.nodes()}
    
    # Initialize distances
    dist = {node: float('-inf') for node in G.nodes()}
    dist[0] = weights[0]  # Start with weight of node 0
    
    # Initialize predecessor tracking - store all predecessors that lead to max weight
    pred = {node: set() for node in G.nodes()}
    
    # Dynamic programming to find path with maximum accumulated weight
    for node in topo_order:
        if dist[node] != float('-inf'):  # If we've reached this node
            for neighbor in G.successors(node):
                if dist[node] + weights[neighbor] > dist[neighbor]:
                    # Found a better path - clear old predecessors and add the new one
                    dist[neighbor] = dist[node] + weights[neighbor]
                    pred[neighbor] = {node}
                elif dist[node] + weights[neighbor] == dist[neighbor]:
                    # Found an equally good path - add this predecessor too
                    pred[neighbor].add(node)
    
    # If there's no path to node 1
    if dist[1] == float('-inf'):
        return [], 0
    
    # Reconstruct all paths with maximum weight
    max_weight = dist[1]
    
    # Get all critical paths using recursive DFS
    def get_all_paths(node):
        if node == 0:
            return [[0]]
        
        all_paths = []
        for p in pred[node]:
            for path in get_all_paths(p):
                all_paths.append(path + [node])
        return all_paths
    
    critical_paths = get_all_paths(1)
    
    return critical_paths, max_weight

# Replace the old find_critical_path function or use this as a new function
def find_critical_path(G):
    """
    Wrapper for backward compatibility - returns the first critical path found.
    """
    paths, weight = find_all_critical_paths(G)
    if paths:
        return paths[0], weight
    return None, 0

def assign_random_weights(G, min_val=0, max_val=15):
    """
    Assign random weights to nodes in the graph.
    """
    weights = {node: random.randint(min_val, max_val) for node in G.nodes()}
    nx.set_node_attributes(G, weights, 'weight')
    return G

def assign_weights_from_dict(G, weights_dict):
    """
    Assign weights to nodes in the graph from a dictionary.
    """
    for node, weight in weights_dict.items():
        if node in G.nodes():
            G.nodes[node]['weight'] = weight
    return G

class InteractiveDAGVisualizerApp:
    def __init__(self, root, patterns, file_path=None, weights=None, weights_file_path=None):
        self.root = root
        self.patterns = patterns  # list of pattern edge lists
        self.file_path = file_path
        self.weights_file_path = weights_file_path  # Path to weights file
        self.all_weights = weights  # List of weight dictionaries for all patterns
        self.current_pattern_idx = 0
        self.graphs = []  # Store NetworkX graph objects
        self.current_layout = "dot"  # Track the current layout
        self.weight_range = (0, 15)  # Default range for random weights
        # self.edge_style = "diagonal"  # New attribute to track edge style
        self.edge_style = "orthogonal"  # New attribute to track edge style
        
        # Interactive state tracking
        self.hover_node = None  # Currently hovered node
        self.dragging = False   # Whether a node is being dragged
        self.drag_node = None   # Node being dragged
        self.last_click_pos = None  # Last mouse position during drag
        self.panning = False    # Whether we're panning the canvas
        self.pan_start = None   # Start position for panning
        self.critical_path_colors = {}  # Map path index to color
        self.critical_path_labels = []  # Store the path labels for hover events
        self.highlighted_path = None  # Currently highlighted critical path
        
        # Add weight editing state
        self.editing_weight = False
        self.edit_node = None
        self.weight_entry = None
        
        # Create graph objects for all patterns
        for i, pattern in enumerate(patterns):
            G = nx.DiGraph()
            for source, dest in pattern:
                G.add_edge(source, dest)
            
            # Assign weights from provided weights or generate random
            if self.all_weights and i < len(self.all_weights):
                assign_weights_from_dict(G, self.all_weights[i])
            else:
                assign_random_weights(G, *self.weight_range)
                
            self.graphs.append(G)

        # Register window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.root.title("Interactive DAG Visualizer")
        self.root.geometry("1000x800")

        # Control frame at the top
        self.control_frame = ttk.Frame(root, padding="10")
        self.control_frame.pack(fill=tk.X)
        
        # Using pattern from file
        ttk.Label(self.control_frame, text="File:").pack(side=tk.LEFT, padx=(0,1))
        file_name = os.path.basename(file_path) if file_path else "No file selected"
        ttk.Label(self.control_frame, text=file_name).pack(side=tk.LEFT)

        # Weight file info
        ttk.Label(self.control_frame, text=" | Weights:").pack(side=tk.LEFT, padx=(10,1))
        weight_file = os.path.basename(weights_file_path) if weights_file_path else "No weights file"
        self.weight_file_label = ttk.Label(self.control_frame, text=weight_file)
        self.weight_file_label.pack(side=tk.LEFT)

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
        
        # Weight control frame (add this new section)
        self.weight_frame = ttk.Frame(root, padding="5")
        self.weight_frame.pack(fill=tk.X)
        
        ttk.Label(self.weight_frame, text="Node Weights:").pack(side=tk.LEFT, padx=(10,5))
        
        # Range for weights
        ttk.Label(self.weight_frame, text="Min:").pack(side=tk.LEFT, padx=(5,1))
        self.min_weight_var = tk.StringVar(value=str(self.weight_range[0]))
        self.min_weight_entry = ttk.Entry(self.weight_frame, textvariable=self.min_weight_var, width=3)
        self.min_weight_entry.pack(side=tk.LEFT)
        
        ttk.Label(self.weight_frame, text="Max:").pack(side=tk.LEFT, padx=(5,1))
        self.max_weight_var = tk.StringVar(value=str(self.weight_range[1]))
        self.max_weight_entry = ttk.Entry(self.weight_frame, textvariable=self.max_weight_var, width=3)
        self.max_weight_entry.pack(side=tk.LEFT)
        
        # Button to shuffle weights
        self.shuffle_button = ttk.Button(self.weight_frame, text="Shuffle Weights", 
                                         command=self.shuffle_weights)
        self.shuffle_button.pack(side=tk.LEFT, padx=10)
        
        # Button to save weights
        self.save_weights_button = ttk.Button(self.weight_frame, text="Save All Weights", 
                                              command=self.save_all_weights)
        self.save_weights_button.pack(side=tk.LEFT, padx=10)
        
        # Toggle to show weights as labels
        self.show_weights_var = tk.BooleanVar(value=True)
        self.show_weights_check = ttk.Checkbutton(
            self.weight_frame, 
            text="Show Weights", 
            variable=self.show_weights_var,
            command=self.update_plot
        )
        self.show_weights_check.pack(side=tk.LEFT, padx=5)
        
        # Weight sum display
        self.weight_sum_label = ttk.Label(self.weight_frame, text="Critical Path Delay: 0")
        self.weight_sum_label.pack(side=tk.LEFT, padx=(20,5))
        
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
        self.layout_selector.bind("<<ComboboxSelected>>", self.change_layout)
        
        # Reset view button
        self.reset_view_button = ttk.Button(self.control_frame, text="Reset View", command=self.reset_view)
        self.reset_view_button.pack(side=tk.LEFT, padx=10)
        
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
        self.path_label_frame = ttk.Frame(self.info_frame)
        self.path_label_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.path_header = ttk.Label(self.path_label_frame, text="Critical Path: None")
        self.path_header.pack(side=tk.TOP, anchor=tk.W)
        
        self.path_container = ttk.Frame(self.path_label_frame)
        self.path_container.pack(side=tk.TOP, fill=tk.X, expand=True)
        
        # Hover info frame (for when user hovers over a node)
        self.hover_frame = ttk.Frame(root, padding="5")
        self.hover_frame.pack(fill=tk.X)
        
        # Hover info label
        self.hover_label = ttk.Label(self.hover_frame, text="Hover over a node to see connections | Drag nodes using middle click to reposition | Use left/right click for pan/zoom")
        self.hover_label.pack(side=tk.LEFT, padx=(0,20))
        
        # Create matplotlib figure and canvas
        self.fig, self.ax = plt.subplots(figsize=(10,8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add the navigation toolbar
        self.toolbar_frame = ttk.Frame(root)
        self.toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()
        
        # Connect event handlers
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        # Enable mouse pan/zoom in the navigation toolbar (default mode)
        self.toolbar.pan()
        
        # Display the first pattern
        self.update_plot()
        self.update_button_states()
        
        # Add edge style toggle
        self.edge_style_var = tk.StringVar(value=self.edge_style)
        self.edge_style_label = ttk.Label(self.weight_frame, text="Edge Style:")
        self.edge_style_label.pack(side=tk.LEFT, padx=(20,5))
        
        self.edge_style_combo = ttk.Combobox(
            self.weight_frame,
            textvariable=self.edge_style_var,
            values=["orthogonal", "diagonal"],
            width=10,
            state="readonly"
        )
        self.edge_style_combo.pack(side=tk.LEFT)
        self.edge_style_combo.bind("<<ComboboxSelected>>", self.change_edge_style)
    
    def change_layout(self, event=None):
        """Handle layout change events."""
        new_layout = self.layout_var.get()
        if new_layout != self.current_layout:
            self.current_layout = new_layout
            
            # Reset the positions for the current graph
            G = self.graphs[self.current_pattern_idx]
            
            # Clear stored positions
            for node in G.nodes():
                if 'pos' in G.nodes[node]:
                    del G.nodes[node]['pos']
            
            # Update the plot with the new layout
            self.update_plot()
        
    def hierarchical_layout(self, G, horizontal=True):
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
        
    def on_click(self, event):
        """Handle mouse click events to start node dragging"""
        if event.inaxes != self.ax or not hasattr(self, 'node_collection'):
            return
            
        # Store toolbar mode before potentially changing it
        current_mode = self.toolbar.mode
            
        # Check if mouse is over any node
        contains, ind = self.node_collection.contains(event)
        
        if contains:
            # Get the index of the node under the mouse
            node_idx = ind['ind'][0] if len(ind['ind']) > 0 else None
            
            if node_idx is not None and node_idx < len(self.current_nodes):
                node = self.current_nodes[node_idx]
                
                # Right-click for editing weight
                if event.button == 3:  # Right mouse button
                    # Disable any active toolbar modes
                    if current_mode:
                        self.toolbar.pan()
                        self.toolbar.pan()  # Toggle twice to disable
                    
                    self.start_weight_editing(node)
                    return
                
                # Middle-click for dragging node
                # Disable any active toolbar modes for node dragging
                if current_mode:
                    self.toolbar.pan()
                    self.toolbar.pan()  # Toggle twice to disable
                
                self.dragging = True
                self.drag_node = node
                self.last_click_pos = (event.xdata, event.ydata)
                
                # Set the cursor to indicate dragging
                self.canvas.get_tk_widget().configure(cursor="fleur")
    
    def start_weight_editing(self, node):
        """Start editing the weight of a node"""
        if self.editing_weight:
            self.cancel_weight_editing()
            
        G = self.graphs[self.current_pattern_idx]
        current_weight = G.nodes[node].get('weight', 1)
        
        # Get node position in screen coordinates
        pos = nx.get_node_attributes(G, 'pos')[node]
        bbox = self.ax.transData.transform(pos)
        
        # Create a text entry widget at the node's position
        self.edit_node = node
        self.editing_weight = True
        
        # Create a small frame for the entry
        edit_frame = ttk.Frame(self.canvas.get_tk_widget())
        
        # Create the entry widget
        entry = ttk.Entry(edit_frame, width=5)
        entry.insert(0, str(current_weight))
        entry.pack(padx=2, pady=2)
        entry.focus_set()
        
        # Position it over the node
        canvas_x, canvas_y = self.canvas.get_tk_widget().winfo_rootx(), self.canvas.get_tk_widget().winfo_rooty()
        window = self.canvas.get_tk_widget().winfo_toplevel()
        window_x, window_y = window.winfo_x(), window.winfo_y()
        
        # Position the entry directly over the node
        x_pos = canvas_x + bbox[0] - 15  # Adjust for entry size
        y_pos = canvas_y + bbox[1] - 10
        
        edit_frame.place(x=bbox[0]-15, y=bbox[1]-10)
        
        self.weight_entry = entry
        
        # Bind events to handle weight editing
        entry.bind("<Return>", self.apply_weight_edit)
        entry.bind("<Escape>", self.cancel_weight_editing)
        entry.bind("<FocusOut>", self.cancel_weight_editing)
        
    def apply_weight_edit(self, event=None):
        """Apply the weight edit to the node"""
        if not self.editing_weight or self.edit_node is None:
            return
            
        try:
            # Get the new weight from the entry
            new_weight = int(self.weight_entry.get())
            
            # Ensure weight is within the allowed range
            min_val, max_val = self.weight_range
            if new_weight < min_val:
                new_weight = min_val
            elif new_weight > max_val:
                new_weight = max_val
                
            # Apply the new weight
            G = self.graphs[self.current_pattern_idx]
            G.nodes[self.edit_node]['weight'] = new_weight
            
            # Clean up the editing interface
            if self.weight_entry and self.weight_entry.winfo_exists():
                self.weight_entry.master.destroy()
                
            self.editing_weight = False
            self.weight_entry = None
            self.edit_node = None
            
            # Update the graph display
            self.update_plot()
            
        except (ValueError, TypeError):
            # If the input is not a valid integer, cancel editing
            self.cancel_weight_editing()
    
    def cancel_weight_editing(self, event=None):
        """Cancel the weight editing and restore the display"""
        if self.weight_entry and self.weight_entry.winfo_exists():
            self.weight_entry.master.destroy()
            
        self.editing_weight = False
        self.weight_entry = None
        self.edit_node = None
        
        # Update the plot to restore the hover state if needed
        if self.hover_node is not None:
            self.highlight_connected_nodes(self.hover_node)
        else:
            self.update_plot(no_draw=True)
            self.canvas.draw()

    def on_motion(self, event):
        """Handle mouse movement for node dragging and hovering"""
        # Don't process motion events if we're editing a weight
        if self.editing_weight:
            return
            
        if not hasattr(self, 'node_collection'):
            return
            
        G = self.graphs[self.current_pattern_idx]
        
        # If dragging a node, update its position
        if self.dragging and self.drag_node is not None and event.xdata is not None and event.ydata is not None:
            # Get the node positions
            pos = nx.get_node_attributes(G, 'pos')
            
            # Calculate the mouse movement delta
            dx = event.xdata - self.last_click_pos[0]
            dy = event.ydata - self.last_click_pos[1]
            
            # Update node position
            old_pos = pos[self.drag_node]
            new_pos = (old_pos[0] + dx, old_pos[1] + dy)
            pos[self.drag_node] = new_pos
            
            # Update the stored positions
            nx.set_node_attributes(G, pos, 'pos')
            
            # Update the last click position
            self.last_click_pos = (event.xdata, event.ydata)
            
            # Redraw the graph
            self.update_plot(no_draw=True)
            self.canvas.draw()
            return
            
        # Handle manual panning if the toolbar is in pan mode and we have a pan_start
        elif self.panning and self.pan_start and event.xdata is not None and event.ydata is not None:
            # Calculate pan distance
            dx = self.pan_start[0] - event.xdata
            dy = self.pan_start[1] - event.ydata
            
            # Get current axis limits
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            
            # Apply pan to limits
            self.ax.set_xlim(xlim[0] + dx, xlim[1] + dx)
            self.ax.set_ylim(ylim[0] + dy, ylim[1] + dy)
            
            # Update canvas
            self.canvas.draw()
            return
            
        # Handle hover highlighting when not dragging
        if not self.dragging:
            # Find the closest node to the mouse pointer
            closest_node = None
            
            # Check if mouse is over any node
            contains, ind = self.node_collection.contains(event)
            
            if contains:
                # Get the index of the node under the mouse
                node_idx = ind['ind'][0] if len(ind['ind']) > 0 else None
                
                if node_idx is not None and node_idx < len(self.current_nodes):
                    closest_node = self.current_nodes[node_idx]
            
            # If hovering over a different node than before, update the display
            if closest_node != self.hover_node:
                self.hover_node = closest_node
                
                if closest_node is not None:
                    # Find predecessors and successors
                    predecessors = list(G.predecessors(closest_node))
                    successors = list(G.successors(closest_node))
                    
                    # Get node weight
                    weight = G.nodes[closest_node].get('weight', 1)
                    
                    # Update the hover info display
                    info_text = f"Node {closest_node} - Weight: {weight} | "
                    if predecessors:
                        info_text += f"Inputs: {', '.join(map(str, predecessors))} | "
                    if successors:
                        info_text += f"Outputs: {', '.join(map(str, successors))}"
                    
                    # Calculate path lengths
                    paths_from_input = []
                    if closest_node != 0 and nx.has_path(G, 0, closest_node):
                        try:
                            paths = list(nx.all_simple_paths(G, 0, closest_node, cutoff=15))
                            if paths:
                                paths_from_input = [(p, len(p)-1) for p in paths]
                                paths_from_input.sort(key=lambda x: x[1], reverse=True)
                        except:
                            try:
                                path = nx.shortest_path(G, 0, closest_node)
                                paths_from_input = [(path, len(path)-1)]
                            except:
                                pass
                    
                    paths_to_output = []
                    if closest_node != 1 and nx.has_path(G, closest_node, 1):
                        try:
                            paths = list(nx.all_simple_paths(G, closest_node, 1, cutoff=15))
                            if paths:
                                paths_to_output = [(p, len(p)-1) for p in paths]
                                paths_to_output.sort(key=lambda x: x[1], reverse=True)
                        except:
                            try:
                                path = nx.shortest_path(G, closest_node, 1)
                                paths_to_output = [(path, len(path)-1)]
                            except:
                                pass
                    
                    # Update the hover info display
                    info_text = f"Node {closest_node} - "
                    if predecessors:
                        info_text += f"Inputs: {', '.join(map(str, predecessors))} | "
                    if successors:
                        info_text += f"Outputs: {', '.join(map(str, successors))}"
                    
                    if paths_from_input:
                        longest_path, length = paths_from_input[0]
                        info_text += f" | Longest path from input: {' → '.join(map(str, longest_path))} (Length: {length})"
                    
                    if paths_to_output:
                        longest_path, length = paths_to_output[0]
                        info_text += f" | Longest path to output: {' → '.join(map(str, longest_path))} (Length: {length})"
                    
                    self.hover_label.config(text=info_text)
                    
                    # Highlight connected nodes
                    self.highlight_connected_nodes(closest_node)
                else:
                    # Reset to default display
                    self.hover_label.config(text="Hover over a node to see connections | Right-click to edit weight | Drag nodes using middle click | Use left/right click for pan/zoom")
                    self.update_plot(no_draw=True)
                    self.canvas.draw()
    
    def on_release(self, event):
        """Handle mouse release to stop node dragging or panning"""
        # Don't process release events if we're editing a weight
        if self.editing_weight:
            return
            
        cursor_reset = False
        
        if self.dragging:
            self.dragging = False
            self.drag_node = None
            self.last_click_pos = None
            cursor_reset = True
            
            # Update the plot to show final state
            self.update_plot()
        
        if self.panning:
            self.panning = False
            self.pan_start = None
            cursor_reset = True
        
        # Reset the cursor if needed
        if cursor_reset:
            self.canvas.get_tk_widget().configure(cursor="")
    
    def on_scroll(self, event):
        """Handle mouse scroll events for zooming"""
        if event.inaxes != self.ax:
            return
            
        # Don't handle scroll if we're dragging a node
        if self.dragging:
            return
            
        # Get the current x and y limits
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        # Set the zoom scale factor
        scale_factor = 0.9 if event.button == 'up' else 1.1
        
        # Calculate new limits - zoom toward the mouse position
        x_center = event.xdata
        y_center = event.ydata
        
        x_min = x_center - (x_center - xlim[0]) * scale_factor
        x_max = x_center + (xlim[1] - x_center) * scale_factor
        y_min = y_center - (y_center - ylim[0]) * scale_factor
        y_max = y_center + (ylim[1] - y_center) * scale_factor
        
        # Set new limits and redraw
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.canvas.draw()
    
    def highlight_connected_nodes(self, focus_node):
        """Highlight nodes connected to the focus node"""
        G = self.graphs[self.current_pattern_idx]
        critical_paths, _ = find_all_critical_paths(G)
        
        # First, redraw with the standard coloring
        self.update_plot(no_draw=True)
        
        # Find connected nodes
        predecessors = list(G.predecessors(focus_node))
        successors = list(G.successors(focus_node))
        
        # Get the current positions
        pos = nx.get_node_attributes(G, 'pos')
        
        # Get node weight
        weight = G.nodes[focus_node].get('weight', 1)
        
        # Redraw the focused node with a highlight
        nx.draw_networkx_nodes(G, pos,
                              nodelist=[focus_node],
                              node_color='yellow',
                              node_size=700,
                              edgecolors='black',
                              linewidths=2,
                              ax=self.ax)
        
        # Highlight predecessor nodes
        if predecessors:
            nx.draw_networkx_nodes(G, pos,
                                  nodelist=predecessors,
                                  node_color='lightgreen',
                                  node_size=600,
                                  ax=self.ax)
            
            # Highlight edges from predecessors
            pred_edges = [(u, focus_node) for u in predecessors]
            if self.edge_style == "diagonal":
                nx.draw_networkx_edges(G, pos,
                                      edgelist=pred_edges,
                                      edge_color='green',
                                      width=2.0,
                                      arrowsize=15,
                                      ax=self.ax)
            else:
                # Draw orthogonal edges with green color
                for u in predecessors:
                    path = self.create_orthogonal_path(pos[u], pos[focus_node])
                    for j in range(len(path) - 1):
                        x1, y1 = path[j]
                        x2, y2 = path[j+1]
                        self.ax.plot([x1, x2], [y1, y2], color='green', linewidth=2.0, zorder=2)
        
        # Highlight successor nodes
        if successors:
            nx.draw_networkx_nodes(G, pos,
                                  nodelist=successors,
                                  node_color='lightblue',
                                  node_size=600,
                                  ax=self.ax)
            
            # Highlight edges to successors
            succ_edges = [(focus_node, v) for v in successors]
            if self.edge_style == "diagonal":
                nx.draw_networkx_edges(G, pos,
                                      edgelist=succ_edges,
                                      edge_color='blue',
                                      width=2.0,
                                      arrowsize=15,
                                      ax=self.ax)
            else:
                # Draw orthogonal edges with blue color
                for v in successors:
                    path = self.create_orthogonal_path(pos[focus_node], pos[v])
                    for j in range(len(path) - 1):
                        x1, y1 = path[j]
                        x2, y2 = path[j+1]
                        self.ax.plot([x1, x2], [y1, y2], color='blue', linewidth=2.0, zorder=2)
        
        # Add weight label to the focused node
        if self.show_weights_var.get():
            nx.draw_networkx_labels(G, pos, 
                                   labels={focus_node: f"{focus_node}\n({weight})"},
                                   font_weight='normal',
                                   ax=self.ax)
        
        # Redraw the canvas
        self.canvas.draw()

    def update_plot(self, event=None, no_draw=False):
        """Update the plot with the current pattern"""
        # Store current axis limits before clearing if we want to preserve the view
        prev_xlim = self.ax.get_xlim() if hasattr(self.ax, 'get_xlim') else None
        prev_ylim = self.ax.get_ylim() if hasattr(self.ax, 'get_ylim') else None
        
        # Store the currently highlighted path index for later
        prev_highlighted = self.highlighted_path
        
        self.ax.clear()
        
        G = self.graphs[self.current_pattern_idx]
        is_dag = nx.is_directed_acyclic_graph(G)
        
        # Calculate node positions based on selected layout or use stored positions
        pos = nx.get_node_attributes(G, 'pos')
        
        # Generate new positions if:
        # 1. No positions stored yet, or
        # 2. This is not a "no_draw" update (which is typically for hover highlighting)
        if not pos or (not no_draw and self.layout_var.get() != self.current_layout):
            layout_type = self.layout_var.get()
            self.current_layout = layout_type
            
            try:
                # Generate layout based on selected type
                if (layout_type == "dot"):
                    pos = self.hierarchical_layout(G, horizontal=True)
                elif layout_type == "circo":
                    pos = nx.circular_layout(G)
                elif layout_type == "twopi":
                    shells = self.get_node_shells(G)
                    pos = nx.shell_layout(G, shells)
                elif layout_type == "neato":
                    pos = nx.kamada_kawai_layout(G)
                elif layout_type == "fdp":
                    pos = nx.spring_layout(G, k=1.5, iterations=100, seed=42)
                elif layout_type == "sfdp":
                    pos = nx.spectral_layout(G)
                else:
                    pos = nx.spring_layout(G, seed=42)
                    
                # Ensure all nodes have positions
                for node in G.nodes():
                    if node not in pos or pos[node] is None:
                        pos[node] = (np.random.random(), np.random.random())
                        
                # Store positions as node attributes for later use
                nx.set_node_attributes(G, pos, 'pos')
            except Exception as e:
                print(f"Layout error: {str(e)}")
                pos = nx.spring_layout(G, seed=42)
                nx.set_node_attributes(G, pos, 'pos')
        
        # Find ALL critical paths considering node weights
        critical_paths, path_weight = find_all_critical_paths(G)
        self.critical_paths = critical_paths  # Store for later reference
        num_critical_paths = len(critical_paths)
        
        # Get node weights
        weights = {node: G.nodes[node].get('weight', 1) for node in G.nodes()}
        
        # Generate colors for critical paths if needed
        if num_critical_paths > 0 and len(self.critical_path_colors) != num_critical_paths:
            colors = self.generate_distinct_colors(num_critical_paths)
            self.critical_path_colors = {i: colors[i] for i in range(num_critical_paths)}
        
        # Set node colors and sizes based on weights
        node_colors = []
        node_sizes = []
        base_size = 500
        
        # Collect all nodes that appear in any critical path, with their path index
        critical_path_nodes = {}  # Maps node to list of path indices it appears in
        for i, path in enumerate(critical_paths):
            for node in path:
                if node not in critical_path_nodes:
                    critical_path_nodes[node] = []
                critical_path_nodes[node].append(i)
        
        for node in G.nodes():
            weight = weights[node]
            # Size variation based on weight
            size = base_size + (weight - self.weight_range[0]) * 30
            node_sizes.append(size)
            
            if node == 0:  # Input node
                node_colors.append('green')
            elif node == 1:  # Output node
                node_colors.append('red')
            elif node in critical_path_nodes:
                # If highlighted path is set, use that path's color
                if prev_highlighted is not None and prev_highlighted in critical_path_nodes[node]:
                    node_colors.append(self.critical_path_colors[prev_highlighted])
                else:
                    # If node is in multiple paths, use orange as a generic color
                    if len(critical_path_nodes[node]) > 1:
                        node_colors.append('orange')
                    else:
                        # Otherwise use the color of the single path it's in
                        path_idx = critical_path_nodes[node][0]
                        node_colors.append(self.critical_path_colors.get(path_idx, 'orange'))
            else:  # Other nodes
                node_colors.append('skyblue')
        
        # Set edge colors - collect critical path edges with their path indices
        critical_path_edges = {}  # Maps edge to list of path indices it appears in
        for i, path in enumerate(critical_paths):
            for j in range(len(path) - 1):
                edge = (path[j], path[j+1])
                if edge not in critical_path_edges:
                    critical_path_edges[edge] = []
                critical_path_edges[edge].append(i)
        
        edge_colors = []
        highlighted_edges = []
        for u, v in G.edges():
            edge = (u, v)
            if edge in critical_path_edges:
                # If highlighted path is set, use that path's color
                if prev_highlighted is not None and prev_highlighted in critical_path_edges[edge]:
                    edge_colors.append(self.critical_path_colors[prev_highlighted])
                    highlighted_edges.append(edge)
                else:
                    # If edge is in multiple paths, use red as a generic color
                    if len(critical_path_edges[edge]) > 1:
                        edge_colors.append('red')
                    else:
                        # Otherwise use the color of the single path it's in
                        path_idx = critical_path_edges[edge][0]
                        edge_colors.append(self.critical_path_colors.get(path_idx, 'red'))
            else:
                edge_colors.append('gray')
        
        # Draw nodes with sizes and colors
        self.node_collection = nx.draw_networkx_nodes(G, pos, 
                                                   node_color=node_colors, 
                                                   node_size=node_sizes, 
                                                   ax=self.ax)
        
        # Store current nodes list for interaction
        self.current_nodes = list(G.nodes())
        
        # Draw edges with selected style
        self.draw_edges(G, pos, edge_colors=edge_colors, highlighted_edges=highlighted_edges, ax=self.ax)
        
        # Draw labels, including weights if enabled
        if self.show_weights_var.get():
            labels = {node: f"{node}\n({weights[node]})" for node in G.nodes()}
        else:
            labels = {node: f"{node}" for node in G.nodes()}
        
        nx.draw_networkx_labels(G, pos, labels=labels, font_weight='normal', ax=self.ax)
        
        # Set title and remove axis
        edge_style_text = "Orthogonal" if self.edge_style == "orthogonal" else "Diagonal" 
        self.ax.set_title(f"Pattern {self.current_pattern_idx} - {self.current_layout.capitalize()} Layout ({edge_style_text} Edges)")
        self.ax.axis('off')
        
        # Restore previous view limits if available and preserving view
        if prev_xlim and prev_ylim and no_draw:
            self.ax.set_xlim(prev_xlim)
            self.ax.set_ylim(prev_ylim)
        
        # Update info labels
        self.index_label.config(text=f"Pattern {self.current_pattern_idx}/{len(self.patterns)-1}")
        self.node_label.config(text=f"Nodes: {G.number_of_nodes()}")
        self.edge_label.config(text=f"Edges: {G.number_of_edges()}")
        self.dag_label.config(text=f"Is DAG: {'Yes' if is_dag else 'No'}")
        
        # Update critical path display with interactive labels
        self.update_critical_path_display()
        
        # Update weight sum label with count of paths
        if num_critical_paths > 1:
            self.weight_sum_label.config(text=f"Critical Path Delay: {path_weight} ({num_critical_paths} paths)")
        elif num_critical_paths == 1:
            self.weight_sum_label.config(text=f"Critical Path Delay: {path_weight}")
        else:
            self.weight_sum_label.config(text=f"Critical Path Delay: 0")
        
        # Update hover label to include weight editing info
        self.hover_label.config(text="Hover over a node to see connections | Right-click to edit weight | Drag nodes using middle click | Use left/right click for pan/zoom | Hover over path labels to highlight")
        
        # Redraw canvas unless we're just updating for hover highlighting
        if not no_draw:
            self.canvas.draw()
    
    def reset_view(self):
        """Reset the view to the original layout"""
        G = self.graphs[self.current_pattern_idx]
        
        # Clear stored positions
        for node in G.nodes():
            if 'pos' in G.nodes[node]:
                del G.nodes[node]['pos']
        
        # Reset zoom and pan
        self.toolbar.home()
        
        # Redraw with fresh layout
        self.update_plot()
    
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
            "Layout Algorithms:\n\n"
            "• dot: Hierarchical layout - Shows the DAG structure clearly\n"
            "• neato: Force-directed using Kamada-Kawai algorithm\n"
            "• fdp: Enhanced spring layout with stronger repulsion\n"
            "• sfdp: Spectral layout - Uses eigenvectors of graph Laplacian\n"
            "• twopi: Shell layout - Nodes arranged in concentric circles\n"
            "• circo: Circular layout - All nodes arranged in a single circle\n"
            "• spring: Standard spring layout with moderate settings\n\n"
            "Edge Styles:\n"
            "• diagonal: Standard straight lines between nodes\n"
            "• orthogonal: Horizontal and vertical lines only (like a schematic)\n\n"
            "Node Weights:\n"
            "• Node size reflects weight value\n"
            "• Critical path now finds the path with maximum total weight\n"
            "• Shuffle Weights button randomizes weights within specified range\n\n"
            "Interactive Features:\n"
            "• Drag nodes using middle click to reposition them\n"
            "• Use the left/right click to pan and zoom\n"
            "• Click 'Reset View' to restore the original layout\n"
            "• Hover over nodes to see connection information"
        )
        messagebox.showinfo("Layout & Interactive Features", layout_info)

    def save_all_weights(self):
        """Save all weights for all patterns to the weights file"""
        if not self.weights_file_path:
            self.weights_file_path = "input_weights.txt"
        
        # Collect current weights from all graphs
        all_weights = []
        for i, G in enumerate(self.graphs):
            weights = {}
            # Ensure we have weights for all nodes 0-15
            for node_idx in range(16):
                # If the node exists in the graph, use its weight
                if node_idx in G.nodes():
                    weights[node_idx] = G.nodes[node_idx].get('weight', random.randint(*self.weight_range))
                else:
                    # If the node doesn't exist in this pattern, use a random weight
                    weights[node_idx] = random.randint(*self.weight_range)
            all_weights.append(weights)
        
        # Write weights to file
        try:
            write_weights(self.weights_file_path, all_weights, len(self.patterns))
            messagebox.showinfo("Success", f"Weights saved to {self.weights_file_path}")
            
            # Update the weight file label
            weight_file = os.path.basename(self.weights_file_path)
            self.weight_file_label.config(text=weight_file)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save weights: {str(e)}")
    
    def shuffle_weights(self):
        """Reassign random weights to the current graph"""
        try:
            # Get min and max values from entries
            min_val = int(self.min_weight_var.get())
            max_val = int(self.max_weight_var.get())
            
            # Ensure min <= max
            if min_val > max_val:
                min_val, max_val = max_val, min_val
                self.min_weight_var.set(str(min_val))
                self.max_weight_var.set(str(max_val))
            
            # Update weight range
            self.weight_range = (min_val, max_val)
            
            # Reassign weights to the current graph
            G = self.graphs[self.current_pattern_idx]
            assign_random_weights(G, min_val, max_val)
            
            # Update the plot
            self.update_plot()
            
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid integer values for min and max weights.")
    
    def generate_distinct_colors(self, n):
        """Generate n visually distinct colors for paths"""
        # Basic set of distinct colors
        base_colors = [
            '#FF6666',  # Light Red
            '#66FF66',  # Light Green
            '#6666FF',  # Light Blue
            '#FFEB99',  # Light Yellow
            '#FF99FF',  # Light Magenta
            '#99FFFF',  # Light Cyan
            '#FFB366',  # Light Orange
            '#B366FF',  # Light Purple
            '#66CC66',  # Muted Green
            '#6666CC',  # Muted Blue
            '#CC6666',  # Muted Red
            '#66CCCC',  # Muted Teal
            '#FFB3B3',  # Pale Pink
            '#CC9966',  # Light Brown
            '#CC99CC'   # Light Plum
        ]
        
        # If we need more colors than in the base set
        if n > len(base_colors):
            # Generate additional colors by interpolating
            import colorsys
            
            HSV_tuples = [(x * 1.0 / n, 0.7, 0.7) for x in range(n)]
            RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
            
            # Convert to hex colors
            rgb_colors = ['#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255)) 
                         for r, g, b in RGB_tuples]
            return rgb_colors
        else:
            return base_colors[:n]
    
    def highlight_critical_path(self, path_idx=None):
        """Highlight a specific critical path or remove highlighting if None"""
        if not hasattr(self, 'critical_paths') or not self.critical_paths:
            return
        
        G = self.graphs[self.current_pattern_idx]
        pos = nx.get_node_attributes(G, 'pos')
        weights = {node: G.nodes[node].get('weight', 1) for node in G.nodes()}
        
        # Update the graph with standard coloring first
        self.update_plot(no_draw=True)
        
        if path_idx is not None and 0 <= path_idx < len(self.critical_paths):
            # Store currently highlighted path
            self.highlighted_path = path_idx
            
            # Get the specific path and its color
            path = self.critical_paths[path_idx]
            color = self.critical_path_colors.get(path_idx, 'red')
            
            # Draw nodes along this path with the path's color
            critical_nodes = path
            nx.draw_networkx_nodes(G, pos, 
                                  nodelist=critical_nodes,
                                  node_color=color,
                                  node_size=700,  # Larger size for emphasis
                                  edgecolors='black',
                                  linewidths=2,
                                  alpha=0.8,
                                  ax=self.ax)
            
            # Draw edges along this path with the path's color
            critical_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            
            if self.edge_style == "diagonal":
                # Standard edge drawing
                nx.draw_networkx_edges(G, pos, 
                                      edgelist=critical_edges,
                                      edge_color=color,
                                      width=3.0,  # Thicker line for emphasis
                                      arrowsize=20,
                                      ax=self.ax)
            else:
                # Draw orthogonal edges
                for i in range(len(path)-1):
                    u, v = path[i], path[i+1]
                    ortho_path = self.create_orthogonal_path(pos[u], pos[v])
                    
                    # Draw each segment of the path
                    for j in range(len(ortho_path) - 1):
                        x1, y1 = ortho_path[j]
                        x2, y2 = ortho_path[j+1]
                        self.ax.plot([x1, x2], [y1, y2], color=color, linewidth=3.0, zorder=3)
            
            # Draw labels for nodes on this path
            if self.show_weights_var.get():
                labels = {node: f"{node}\n({weights[node]})" for node in critical_nodes}
            else:
                labels = {node: f"{node}" for node in critical_nodes}
                
            nx.draw_networkx_labels(G, pos, 
                                   labels=labels,
                                   font_weight='normal',
                                   font_size=12,
                                   ax=self.ax)
        
        self.canvas.draw()
    
    def create_orthogonal_path(self, start_pos, end_pos, edge_index=0, total_edges=1):
        """Create an orthogonal path between two points with offset to avoid overlaps."""
        # Extract coordinates
        x1, y1 = start_pos
        x2, y2 = end_pos
        
        # Calculate a unique offset for this edge
        # Use edge_index to create different paths for different edges
        offset_factor = 0.03  # Base offset amount
        
        # If we have multiple edges, distribute them
        if total_edges > 1:
            # Convert edge_index to a normalized offset (-0.5 to 0.5)
            normalized_offset = (edge_index / (total_edges - 1) - 0.5) if total_edges > 1 else 0
            offset = normalized_offset * offset_factor * (total_edges + 1)
        else:
            offset = 0
        
        # Determine if horizontal or vertical distance is greater
        if abs(x2 - x1) > abs(y2 - y1):
            # Horizontal dominant path
            # Calculate the midpoint with offset
            mid_x = (x1 + x2) / 2
            
            # Create a path with an offset in Y direction at the midpoint
            if y1 != y2:  # Only add offset if there's a vertical component
                path = [(x1, y1), 
                       (mid_x, y1 + offset), 
                       (mid_x, y2 - offset), 
                       (x2, y2)]
            else:
                # Direct horizontal path with a slight bend to avoid overlaps
                mid_y = y1 + offset
                path = [(x1, y1), 
                       (x1 + (x2-x1)/4, mid_y), 
                       (x1 + 3*(x2-x1)/4, mid_y), 
                       (x2, y2)]
        else:
            # Vertical dominant path
            # Calculate the midpoint with offset
            mid_y = (y1 + y2) / 2
            
            # Create a path with an offset in X direction at the midpoint
            if x1 != x2:  # Only add offset if there's a horizontal component
                path = [(x1, y1), 
                       (x1 + offset, mid_y), 
                       (x2 - offset, mid_y), 
                       (x2, y2)]
            else:
                # Direct vertical path with a slight bend to avoid overlaps
                mid_x = x1 + offset
                path = [(x1, y1), 
                       (mid_x, y1 + (y2-y1)/4), 
                       (mid_x, y1 + 3*(y2-y1)/4), 
                       (x2, y2)]
                
        return path
    
    def draw_edges(self, G, pos, edge_colors=None, highlighted_edges=None, ax=None):
        """Draw edges using the current edge style with improved orthogonal routing."""
        if ax is None:
            ax = self.ax
            
        if edge_colors is None:
            edge_colors = ['gray'] * len(G.edges())
            
        if highlighted_edges is None:
            highlighted_edges = []
            
        if self.edge_style == "diagonal":
            # Standard straight-line edges
            nx.draw_networkx_edges(G, pos, 
                                 edge_color=edge_colors, 
                                 width=1.5, 
                                 arrowsize=15,
                                 ax=ax)
        else:
            # Orthogonal edges with better routing
            edge_list = list(G.edges())
            
            # Group edges by their source and target for better layout
            # This helps us know when multiple edges connect the same pairs of nodes
            edge_groups = {}
            for i, (u, v) in enumerate(edge_list):
                # Create a key that uniquely identifies source-target node pairs
                src, tgt = pos[u], pos[v]
                key = (round(src[0], 3), round(src[1], 3), round(tgt[0], 3), round(tgt[1], 3))
                
                if key not in edge_groups:
                    edge_groups[key] = []
                edge_groups[key].append((i, u, v))
            
            # Draw each edge with appropriate offset based on its group
            for key, edges in edge_groups.items():
                for j, (i, u, v) in enumerate(edges):
                    # Get edge color and width
                    color = edge_colors[i] if i < len(edge_colors) else 'gray'
                    width = 2.0 if (u, v) in highlighted_edges else 1.5
                    
                    # Create orthogonal path with index-based offset to avoid overlaps
                    path = self.create_orthogonal_path(pos[u], pos[v], j, len(edges))
                    
                    # Draw path segments
                    for j in range(len(path) - 1):
                        x1, y1 = path[j]
                        x2, y2 = path[j+1]
                        ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, zorder=1)
                    
                    # Add arrow at the end
                    self.draw_edge_arrow(path[-2], path[-1], color, ax)
    
    def draw_edge_arrow(self, start_pos, end_pos, color, ax):
        """Draw an arrow at the end of an edge segment."""
        # Calculate direction vector
        x1, y1 = start_pos
        x2, y2 = end_pos
        dx, dy = x2 - x1, y2 - y1
        
        # If the segment is too short, don't draw an arrow
        length = np.sqrt(dx**2 + dy**2)
        if length < 0.01:
            return
            
        # Normalize direction
        dx, dy = dx / length, dy / length
        
        # Position the arrow slightly before the end point
        arrow_pos = (x2 - 0.02 * dx, y2 - 0.02 * dy)
        
        # Draw the arrow with appropriate direction
        ax.arrow(arrow_pos[0], arrow_pos[1], 0.02 * dx, 0.02 * dy, 
               head_width=0.015, head_length=0.02, fc=color, ec=color)

    def on_path_label_enter(self, event, path_idx):
        """Handle mouse entering a path label"""
        self.highlight_critical_path(path_idx)
    
    def on_path_label_leave(self, event):
        """Handle mouse leaving a path label"""
        self.highlight_critical_path(None)  # Remove path highlighting
        self.highlighted_path = None
    
    def update_critical_path_display(self):
        """Update the critical path display with interactive labels"""
        # Clear previous labels
        for widget in self.path_container.winfo_children():
            widget.destroy()
        self.critical_path_labels = []
        
        if not hasattr(self, 'critical_paths') or not self.critical_paths:
            self.path_header.config(text="Critical Path: None")
            return
            
        G = self.graphs[self.current_pattern_idx]
        weights = {node: G.nodes[node].get('weight', 1) for node in G.nodes()}
        num_paths = len(self.critical_paths)
        
        # Generate colors for paths if not already done
        if len(self.critical_path_colors) != num_paths:
            colors = self.generate_distinct_colors(num_paths)
            self.critical_path_colors = {i: colors[i] for i in range(num_paths)}
        
        # Update the header
        if num_paths == 1:
            self.path_header.config(text="Critical Path:")
        else:
            self.path_header.config(text=f"Critical Paths ({num_paths}):")
        
        # Create label for each path (up to 10 for space reasons)
        max_to_show = min(10, num_paths)
        for i in range(max_to_show):
            path = self.critical_paths[i]
            path_str = " → ".join([f"{node}({weights[node]})" for node in path])
            
            # Create the label with the path's color
            color = self.critical_path_colors.get(i, 'black')
            label = ttk.Label(self.path_container, 
                           text=f"{i+1}: {path_str}", 
                           foreground=color)
            label.pack(side=tk.TOP, anchor=tk.W, padx=(15, 0))
            
            # Bind hover events
            label.bind("<Enter>", lambda event, idx=i: self.on_path_label_enter(event, idx))
            label.bind("<Leave>", self.on_path_label_leave)
            
            self.critical_path_labels.append(label)
        
        # If there are more paths than we're showing
        if num_paths > max_to_show:
            ttk.Label(self.path_container, 
                   text=f"... and {num_paths - max_to_show} more").pack(
                   side=tk.TOP, anchor=tk.W, padx=(15, 0))

    def change_edge_style(self, event=None):
        """Handle edge style change events."""
        new_style = self.edge_style_var.get()
        if new_style != self.edge_style:
            self.edge_style = new_style
            self.update_plot()

def main():
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="Interactive DAG Visualizer")
    parser.add_argument("file_path", help="Path to the input file with DAG patterns")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of patterns to process")
    parser.add_argument("--weights", default="input_weights.txt", help="Path to the node weights file (default: input_weights.txt)")
    
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
        
        # Read or generate weights
        weights = None
        weights_file_path = args.weights
        
        if os.path.exists(weights_file_path):
            print(f"Reading weights from {weights_file_path}...")
            weights = read_weights(weights_file_path)
            
            # Add debug output to check weight loading
            if weights:
                print(f"Successfully loaded {len(weights)} weight sets")
                print(f"Sample weights from first pattern: {list(weights[0].items())[:5]}")
            else:
                print("Failed to load any weights from file")
            
            # Check if weights match patterns
            if weights and len(weights) != len(patterns):
                print(f"Warning: Number of weight sets ({len(weights)}) doesn't match number of patterns ({len(patterns)})")
                print("Generating new random weights...")
                weights = None
        
        # Generate weights if needed
        if not weights:
            print(f"Generating random weights for {len(patterns)} patterns...")
            weights = generate_random_weights(len(patterns), num_nodes=16)
            
            # Write weights to file
            print(f"Saving weights to {weights_file_path}...")
            write_weights(weights_file_path, weights, len(patterns))
        
        # Create and run the GUI
        root = tk.Tk()
        app = InteractiveDAGVisualizerApp(root, patterns, args.file_path, weights, weights_file_path)
        root.mainloop()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

