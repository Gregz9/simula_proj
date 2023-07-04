# Building the graph with nodes = 4

import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph(directed=True)
G.add_node(1)
G.add_node(2)
G.add_node(3)
G.add_node(4)

G.add_edge(1, 2)
G.add_edge(1, 3)
G.add_edge(1, 4)

G.add_edge(2, 1)
G.add_edge(2, 3)
G.add_edge(2, 4)

G.add_edge(3, 1)
G.add_edge(3, 2)
G.add_edge(3, 4)

G.add_edge(4, 1)
G.add_edge(4, 2)
G.add_edge(4, 3)

pos = {1: [0.75, 1.0], 2: [0.75, 0.15], 3: [0.5, -0.5], 4: [1.0, -0.5]}


edge_labels = {
    (1, 2): "$\\phi_{2\\to 1}$\n $\\phi_{1\\to 2}$",
    (1, 3): "$\\phi_{1\\to 3}$\n $\\phi_{3\\to 1}$",
    (1, 4): "$\\phi_{4\\to 1}$\n $\\phi_{1\\to 4}$",
    (2, 3): "$\\phi_{2\\to 3}$\n $\\phi_{3\\to 2}$",
    (2, 4): "$\\phi_{4\\to 2}$\n $\\phi_{2\\to 4}$",
    (3, 4): "$\\phi_{4\\to 3}$\n $\\phi_{3\\to 4}$",
}

fig = plt.figure(1, figsize=(14, 10))
nx.draw(
    G,
    with_labels=True,
    node_color="skyblue",
    edge_cmap=plt.cm.Blues,
    pos=pos,
    connectionstyle="arc3, rad = 0.2",
    node_size=3000,
    arrowsize=14,
    arrowstyle="simple",
    font_size=30,
)

nx.draw_networkx_edge_labels(
    G, pos, edge_labels=edge_labels, font_size=20, bbox=dict(alpha=0)
)

plt.show()
