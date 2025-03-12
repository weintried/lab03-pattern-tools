import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

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

class InteractiveDAGVisualizerApp:
    def layout_selector_changed(self, event):
        """Handle layout selection change"""
        # Only regenerate layout if it actually changed
        G = self.graphs[self.current_pattern_idx]
        current_layout = getattr(G, 'current_layout', None)
        new_layout = self.layout_var.get()
        
        if current_layout != new_layout:
            # Force update with new layout
            self.update_plot(force_layout=True)
            
    def __init__(self, root, patterns, file_path=None):
        self.root = root
        self.patterns = patterns  # list of pattern edge lists
        self.file_path = file_path
        self.current_pattern_idx = 0
        self.graphs = []  # Store NetworkX graph objects
        
        # Interactive state tracking
        self.hover_node = None  # Currently hovered node
        self.dragging = False   # Whether a node is being dragged
        self.drag_node = None   # Node being dragged
        self.last_click_pos = None  # Last mouse position during drag
        self.panning = False    # Whether we're panning the canvas
        self.pan_start = None   # Start position for panning
        
        # Create graph objects for all patterns
        for pattern in patterns:
            G = nx.DiGraph()
            for source, dest in pattern:
                G.add_edge(source, dest)
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
        self.layout_selector.bind("<<ComboboxSelected>>", self.layout_selector_changed)
        
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
        self.path_label = ttk.Label(self.info_frame, text="Critical Path: None")
        self.path_label.pack(side=tk.LEFT, padx=(0,20))
        
        # Hover info frame (for when user hovers over a node)
        self.hover_frame = ttk.Frame(root, padding="5")
        self.hover_frame.pack(fill=tk.X)
        
        # Hover info label
        self.hover_label = ttk.Label(self.hover_frame, text="Hover over a node to see connections | Drag nodes to reposition | Use toolbar for pan/zoom")
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
                # Disable any active toolbar modes for node dragging
                if current_mode:
                    self.toolbar.pan()
                    self.toolbar.pan()  # Toggle twice to disable
                
                self.dragging = True
                self.drag_node = self.current_nodes[node_idx]
                self.last_click_pos = (event.xdata, event.ydata)
                
                # Set the cursor to indicate dragging
                self.canvas.get_tk_widget().configure(cursor="fleur")
    
    def on_motion(self, event):
        """Handle mouse movement for node dragging and hovering"""
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
                    
                    # Calculate path lengths
                    paths_from_input = []
                    if closest_node != 0 and nx.has_path(G, 0, closest_node):
                        try:
                            paths = list(nx.all_simple_paths(G, 0, closest_node, cutoff=10))
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
                            paths = list(nx.all_simple_paths(G, closest_node, 1, cutoff=10))
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
                    self.hover_label.config(text="Hover over a node to see connections | Drag nodes to reposition | Use toolbar for pan/zoom")
                    self.update_plot(no_draw=True)
                    self.canvas.draw()
    
    def on_release(self, event):
        """Handle mouse release to stop node dragging or panning"""
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
        critical_path, _ = find_critical_path(G)
        
        # First, redraw with the standard coloring
        self.update_plot(no_draw=True)
        
        # Find connected nodes
        predecessors = list(G.predecessors(focus_node))
        successors = list(G.successors(focus_node))
        
        # Get the current positions
        pos = nx.get_node_attributes(G, 'pos')
        
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
            edges = [(u, focus_node) for u in predecessors]
            nx.draw_networkx_edges(G, pos,
                                  edgelist=edges,
                                  edge_color='green',
                                  width=2.0,
                                  arrowsize=15,
                                  ax=self.ax)
        
        # Highlight successor nodes
        if successors:
            nx.draw_networkx_nodes(G, pos,
                                  nodelist=successors,
                                  node_color='lightblue',
                                  node_size=600,
                                  ax=self.ax)
            
            # Highlight edges to successors
            edges = [(focus_node, v) for v in successors]
            nx.draw_networkx_edges(G, pos,
                                  edgelist=edges,
                                  edge_color='blue',
                                  width=2.0,
                                  arrowsize=15,
                                  ax=self.ax)
        
        # Redraw the canvas
        self.canvas.draw()

    def update_plot(self, event=None, no_draw=False, force_layout=False):
        """Update the plot with the current pattern"""
        # Store current axis limits before clearing if we want to preserve the view
        prev_xlim = self.ax.get_xlim() if hasattr(self.ax, 'get_xlim') else None
        prev_ylim = self.ax.get_ylim() if hasattr(self.ax, 'get_ylim') else None
        
        self.ax.clear()
        
        G = self.graphs[self.current_pattern_idx]
        is_dag = nx.is_directed_acyclic_graph(G)
        
        # Calculate node positions based on selected layout or use stored positions
        pos = nx.get_node_attributes(G, 'pos')
        layout_status = getattr(G, 'current_layout', 'custom')  # Default layout status
        
        # If positions don't exist, force_layout is True, or the layout has changed, generate new layout
        layout_type = self.layout_var.get()
        if not pos or force_layout or not hasattr(G, 'current_layout') or G.current_layout != layout_type:
            layout_type = self.layout_var.get()
            
            try:
                # Generate layout based on selected type
                if layout_type == "dot":
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
        self.node_collection = nx.draw_networkx_nodes(G, pos, 
                                                    node_color=node_colors, 
                                                    node_size=500, 
                                                    ax=self.ax)
        
        # Store current nodes list for interaction
        self.current_nodes = list(G.nodes())
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, 
                              edge_color=edge_colors, 
                              width=1.5, 
                              arrowsize=15,
                              ax=self.ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_weight='bold', ax=self.ax)
        
        self.ax.set_title(f"Pattern {self.current_pattern_idx} - Layout: {layout_status}")
        self.ax.axis('off')
        
        # Restore previous view limits if available and not doing a full reset
        if prev_xlim and prev_ylim and no_draw:
            self.ax.set_xlim(prev_xlim)
            self.ax.set_ylim(prev_ylim)
        
        # Update info labels
        self.index_label.config(text=f"Pattern {self.current_pattern_idx}/{len(self.patterns)-1}")
        self.node_label.config(text=f"Nodes: {G.number_of_nodes()}")
        self.edge_label.config(text=f"Edges: {G.number_of_edges()}")
        self.dag_label.config(text=f"Is DAG: {'Yes' if is_dag else 'No'}")
        
        if critical_path:
            path_str = " → ".join(map(str, critical_path))
            self.path_label.config(text=f"Critical Path: {path_str} (Length: {path_length})")
        else:
            self.path_label.config(text="Critical Path: None")
        
        # Redraw canvas unless we're just updating for hover highlighting
        if not no_draw:
            self.canvas.draw()
    
    def reset_view(self):
        """Reset the view to the original layout"""
        G = self.graphs[self.current_pattern_idx]
        
        # Clear stored positions
        for node in G.nodes():
            G.nodes[node].pop('pos', None)
        
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
            "Interactive Features:\n"
            "• Drag nodes to reposition them\n"
            "• Use the toolbar to pan and zoom\n"
            "• Click 'Reset View' to restore the original layout\n"
            "• Hover over nodes to see connection information"
        )
        messagebox.showinfo("Layout & Interactive Features", layout_info)

def main():
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="Interactive DAG Visualizer")
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
        app = InteractiveDAGVisualizerApp(root, patterns, args.file_path)
        root.mainloop()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()