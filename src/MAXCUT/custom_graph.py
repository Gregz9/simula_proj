import networkx as nx 
import matplotlib.pyplot as plt
import numpy as np

G = nx.Graph()
n = 4
G.add_nodes_from(np.arange(0,n, 1))

colors = ["b" for node in G.nodes()]

def draw_graph(G, colors, pos): 
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(
            G, node_color=colors, node_size=600, alpha=0.8, ax=default_axes, pos=pos) 

pos = nx.spring_layout(G) 
print(pos)
pos = dict()
pos[0] = np.array([1.0, 2.0])
pos[1] = np.array([3.0, 1.0])
pos[2] = np.array([0.0, 5.0])
pos[3] = np.array([-2.0, 4.0])
print(pos)

draw_graph(G, colors, pos)
plt.show()
