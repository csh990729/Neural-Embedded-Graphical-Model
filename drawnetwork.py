import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.path import Path
import matplotlib.patches as mpatches

def draw_neural_network(neurons_per_layer, 
                        layer_labels=None, 
                        node_size=300, 
                        node_colors=None,
                        edge_alpha=0.4,
                        edge_width=1.5,
                        curved_edges=True,
                        figsize=(12, 8),
                        title="Neural Network Architecture",
                        dark_mode=False):
    """
    Draw an aesthetically pleasing neural network diagram.
    
    Parameters:
        neurons_per_layer : list of int, number of neurons in each layer
        layer_labels : list of str, optional labels for layers (e.g., ['Input', 'Hidden 1', ...])
        node_size : int, size of each node (passed to scatter)
        node_colors : list of matplotlib colors (one per layer) or None for automatic gradient
        edge_alpha : float, transparency of edges (0 to 1)
        edge_width : float, line width of edges
        curved_edges : bool, if True draw Bezier curves, else straight lines
        figsize : tuple, figure size
        title : str
        dark_mode : bool, use a dark background with light elements
    """
    n_layers = len(neurons_per_layer)
    max_neurons = max(neurons_per_layer)
    
    # Set up figure and axes
##    if dark_mode:
##        plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=figsize)
    if dark_mode:
##        ax.set_facecolor('#1a1a1a')
##        fig.patch.set_facecolor('#1a1a1a')
##        edge_color = 'white'
##        ax.set_facecolor('#f5f5f5')
##        fig.patch.set_facecolor('#f5f5f5')
        edge_color = '#2c3e50'
        node_edge_color = 'none'
        text_color = 'white'
    else:
##        ax.set_facecolor('#f5f5f5')
##        fig.patch.set_facecolor('#f5f5f5')
        edge_color = '#2c3e50'
        node_edge_color = 'white'
        text_color = '#2c3e50'
    
    # Node coordinates: x = layer index, y = neuron index (centered)
    x_positions = np.linspace(0, 1, n_layers)
    node_coords = []  # list of lists of (x, y) per layer
    for l, n_neurons in enumerate(neurons_per_layer):
        y_positions = np.linspace(0, 1, n_neurons)
        # Center each layer vertically relative to the max height
        y_centered = y_positions - (y_positions[-1] - y_positions[0])/2 + 0.5
        layer_coords = [(x_positions[l], y) for y in y_centered]
        node_coords.append(layer_coords)
    
    # Determine node colors (per layer gradient)
    if node_colors is None:
        # Create a smooth gradient from blue to orange across layers
        cmap = plt.cm.viridis if dark_mode else plt.cm.plasma
        node_colors = [cmap(i / (n_layers-1)) for i in range(n_layers)]
    else:
        assert len(node_colors) == n_layers, "node_colors must match number of layers"
    
    # Draw edges (between consecutive layers)
    for l in range(n_layers - 1):
        layer_from = node_coords[l]
        layer_to = node_coords[l+1]
        for i, (x1, y1) in enumerate(layer_from):
            for j, (x2, y2) in enumerate(layer_to):
                if curved_edges:
                    # Bezier curve: control points offset vertically
                    mid_x = (x1 + x2) / 2
                    offset = 0.2 * (j - (neurons_per_layer[l+1]-1)/2) / max_neurons
                    verts = [(x1, y1), (mid_x, y1 + offset), (mid_x, y2 + offset), (x2, y2)]
                    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
                    path = Path(verts, codes)
                    patch = mpatches.PathPatch(path, fill=False, 
                                               edgecolor=edge_color, 
                                               alpha=edge_alpha,
                                               linewidth=edge_width)
                    ax.add_patch(patch)
                else:
                    ax.plot([x1, x2], [y1, y2], color=edge_color, 
                            alpha=edge_alpha, linewidth=edge_width)
    
    # Draw nodes
    for l, coords in enumerate(node_coords):
        xs, ys = zip(*coords)
        color = node_colors[l]
        ax.scatter(xs, ys, s=node_size, c=[color], edgecolors=node_edge_color, 
                   linewidth=2, zorder=3, alpha=0.9)
        # Optional: add neuron labels (uncomment if desired)
        # for i, (x, y) in enumerate(coords):
        #     ax.text(x, y, str(i+1), ha='center', va='center', 
        #             fontsize=8, color='black', fontweight='bold')
    
    # Add layer labels
    if layer_labels is not None:
        for l, label in enumerate(layer_labels):
            x = x_positions[l]
            y = -0.08  # below the lowest node
            ax.text(x, y, label, ha='center', va='top', fontsize=12, 
                    color=text_color, fontweight='semibold')
    
    # Adjust plot limits and remove axes
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.2, 1.2)
    ax.axis('off')
    ax.set_title(title, fontsize=16, color=text_color, pad=20)
    
    plt.tight_layout()
    return fig, ax

# Example usage
if __name__ == "__main__":
    # Define architecture: 4 input, 5, 6, 4, 3 output
##    layers = [3, 4, 6, 5, 3]
    layers = [5, 8, 4]
    labels = None
##    labels = ["Input", "Hidden 1", "Hidden 2", "Hidden 3", "Output"]
    
    # Light mode with curved edges
    fig1, _ = draw_neural_network(layers, layer_labels=labels, 
                                   curved_edges=True, dark_mode=False,
                                   title="Feed‑Forward Neural Network (Light)")
    # Dark mode with straight edges
    fig2, _ = draw_neural_network(layers, layer_labels=labels,
                                   curved_edges=False, dark_mode=True,
                                   title="Feed‑Forward Neural Network (Dark)")
    plt.show()
    
    # Save high-resolution image if desired
    # fig1.savefig("nn_light.png", dpi=300, bbox_inches='tight')
    # fig2.savefig("nn_dark.png", dpi=300, bbox_inches='tight')
