"""
Exploration of vispy's GraphVisual with NetworkX layouts.

Testing whether vispy.visuals.graphs.GraphVisual + NetworkxCoordinates
are worth integrating into compneurovis as a connectivity view.

Things covered here:
  - Basic GraphVisual setup in a SceneCanvas
  - NetworkxCoordinates bridging a nx.Graph to vispy layout (static)
  - Built-in force_directed layout for animated settling (ANIMATE=True)
  - Per-node face colors (cell-type coloring)
  - A neuron-like connectivity topology (feedforward chain + recurrent)

Toggle ANIMATE below to switch between static spring and animated force-directed.

Run:
    python scratch/vispy_graph_exploration.py
"""

import numpy as np
import networkx as nx
from vispy import app, scene
from vispy.visuals.graphs import GraphVisual
from vispy.visuals.graphs.layouts import NetworkxCoordinates
from vispy.visuals.graphs.layouts.force_directed import fruchterman_reingold

# Set True for animated force-directed settling; False for static NX spring layout.
ANIMATE = True

# ---------------------------------------------------------------------------
# Build a layered circuit with recurrent inhibition and top-down feedback
# ---------------------------------------------------------------------------
# Inspired loosely by a cortical column: sensory inputs → a recurrent
# excitatory layer gated by local inhibitory interneurons → output layer,
# with top-down feedback and a global neuromodulator.
#
# Node types: 0=sensory (orange), 1=excitatory (blue), 2=inhibitory (red),
#             3=modulatory (green)
G = nx.DiGraph()

# 0,1,2 — sensory inputs
# 3,4,5 — first-layer excitatory interneurons (recurrent)
# 6,7   — inhibitory interneurons
# 8,9,10 — second-layer / output excitatory neurons (recurrent)
# 11    — neuromodulator
node_types = {
    0: 0, 1: 0, 2: 0,
    3: 1, 4: 1, 5: 1,
    6: 2, 7: 2,
    8: 1, 9: 1, 10: 1,
    11: 3,
}

# Sensory → first layer (convergent; node 4 is the main target)
G.add_edges_from([(0, 3), (0, 4), (1, 4), (2, 4), (2, 5)])

# First layer — dense recurrent excitation
G.add_edges_from([(3, 4), (4, 3), (4, 5), (5, 4), (3, 5)])

# First layer → inhibitory interneurons
G.add_edges_from([(3, 6), (4, 6), (4, 7), (5, 7)])

# Inhibitory feedback onto first layer (lateral inhibition)
G.add_edges_from([(6, 3), (6, 4), (7, 4), (7, 5)])

# Inhibitory mutual coupling (disinhibition loop)
G.add_edges_from([(6, 7), (7, 6)])

# First layer → second layer (divergent from hub node 4)
G.add_edges_from([(3, 8), (4, 8), (4, 9), (5, 9), (4, 10), (5, 10)])

# Second layer — recurrent excitation
G.add_edges_from([(8, 9), (9, 8), (9, 10), (10, 9)])

# Top-down feedback: second layer → first layer
G.add_edges_from([(9, 4), (10, 5)])

# Neuromodulator broadcasts to both layers and inhibitory interneurons
G.add_edges_from([(11, 3), (11, 4), (11, 7), (11, 9)])

# ---------------------------------------------------------------------------
# Compute layout
# ---------------------------------------------------------------------------
# Two modes:
#   ANIMATE=False — NetworkxCoordinates wraps any nx layout and yields once
#                   (static). Layout string maps to nx.<name>_layout.
#                   Other options: "circular", "kamada_kawai", "shell".
#   ANIMATE=True  — vispy's built-in "force_directed" layout is iterative;
#                   it yields updated positions each frame as physics settles.
adjacency_mat = nx.to_numpy_array(G, dtype=np.float32)

if ANIMATE:
    # Seed FR from NX spring positions so nodes start in sensible positions
    # instead of fully random — avoids the chaotic high-temperature phase.
    nx_pos = nx.spring_layout(G, seed=42)
    init_pos = np.array([nx_pos[i] for i in range(len(G))], dtype=np.float32)
    init_pos = (init_pos - init_pos.min()) / (init_pos.max() - init_pos.min())
    # More iterations → slower temperature drop → less overshoot/oscillation.
    layout = fruchterman_reingold(iterations=250, pos=init_pos)
else:
    layout = NetworkxCoordinates(G, layout="spring", seed=42)
    adjacency_mat = layout.adj

# ---------------------------------------------------------------------------
# Per-node colors based on cell type
# ---------------------------------------------------------------------------
TYPE_COLORS = {
    0: (0.95, 0.60, 0.15, 1.0),   # sensory    — orange
    1: (0.25, 0.55, 0.95, 1.0),   # excitatory — blue
    2: (0.9,  0.25, 0.25, 1.0),   # inhibitory — red
    3: (0.25, 0.85, 0.45, 1.0),   # modulatory — green
}
n_nodes = len(G)
face_colors = np.array([TYPE_COLORS[node_types[i]] for i in range(n_nodes)], dtype=np.float32)

# ---------------------------------------------------------------------------
# vispy scene setup
# ---------------------------------------------------------------------------
# GraphVisual is a CompoundVisual (ArrowVisual + MarkersVisual). To use it in
# a SceneCanvas it must be wrapped into a scene node via create_visual_node().
SceneGraph = scene.visuals.create_visual_node(GraphVisual)

canvas = scene.SceneCanvas(title="vispy graph exploration", bgcolor="#1a1a2e", size=(900, 700), show=True)
view = canvas.central_widget.add_view()

# PanZoom is the right camera for 2D graph views
view.camera = scene.cameras.PanZoomCamera(aspect=1)

# Graph data is normalized to [0,1] by NetworkxCoordinates, so set the camera
# to show a slightly padded range.
view.camera.set_range(x=(-0.1, 1.1), y=(-0.1, 1.1))

# ---------------------------------------------------------------------------
# Add graph visual
# ---------------------------------------------------------------------------
# adjacency_mat: NetworkxCoordinates.adj returns the nx sparse adjacency matrix.
# layout: the NetworkxCoordinates instance itself is callable — GraphVisual
#   calls layout(adjacency_mat, directed) which yields (node_pos, edge_pos, arrows).
graph_node = SceneGraph(
    adjacency_mat=adjacency_mat,
    directed=True,
    layout=layout,
    animate=ANIMATE,
    node_size=18,
    face_color=face_colors,
    border_color="white",
    border_width=1.5,
    line_color=(0.6, 0.6, 0.6, 0.7),
    line_width=2,
    arrow_type="stealth",
    arrow_size=10,
    parent=view.scene,
)

if ANIMATE:
    # Drive layout animation via a timer. Each tick advances one force step
    # and queues a canvas redraw. animate_layout() returns True when settled.
    def _tick(_event):
        done = graph_node.animate_layout()
        canvas.update()
        if done:
            timer.stop()

    timer = app.Timer(interval=0.05, connect=_tick, start=True)

# ---------------------------------------------------------------------------
# Simple text legend
# ---------------------------------------------------------------------------
legend_items = [
    ("sensory",     TYPE_COLORS[0]),
    ("excitatory",  TYPE_COLORS[1]),
    ("inhibitory",  TYPE_COLORS[2]),
    ("modulatory",  TYPE_COLORS[3]),
]
for i, (label, color) in enumerate(legend_items):
    # parent=canvas.scene puts text in pixel space (origin top-left, y-down),
    # so it stays fixed when the graph view is panned/zoomed.
    scene.visuals.Text(
        text=label,
        pos=(14, 18 + i * 26),
        color=color,
        font_size=9,
        face="OpenSans",
        italic=True,
        anchor_x="left",
        anchor_y="center",
        parent=canvas.scene,
    )

# ---------------------------------------------------------------------------
# Notes on what to explore next
# ---------------------------------------------------------------------------
# - Try layout="circular" or layout="kamada_kawai" to compare
# - face_color accepts per-node arrays, so it can be updated live for
#   activity-based coloring (e.g. brighter = higher firing rate)
# - For live updates, call graph_node._node.set_data(face_color=new_colors)
#   each frame — avoids rebuilding the adjacency/layout
# - GraphVisual.animate=True + canvas.on_draw using animate_layout() for
#   force-directed settling animation

if __name__ == "__main__":
    app.run()
