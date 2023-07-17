import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math


def calculate_angle(point1, point2, offset=0):
    """Calculate angle between two points with respect to x-axis.

    Args:
        point1 (tuple): Point 1 with x and y coordinates.
        point2 (tuple): Point 2 with x and y coordinates.
        offset (int, optional): Angle offset in degrees. Default is 0.

    Returns:
        float: Angle in degrees between the line from point1 to point2 and the x-axis.
    """
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    angle = -math.degrees(math.atan2(dy, dx))
    return (angle + offset) % 360


def get_euclidean_distance(point1, point2):
    """Get Euclidean distance between two points.

    Args:
        point1 (tuple): Point 1 with x and y coordinates.
        point2 (tuple): Point 2 with x and y coordinates.

    Returns:
        float: Euclidean distance between point 1 and point 2.
    """
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def calculate_distances(centres: list) -> np.ndarray:
    """Calculate distances between each pair of centers.

    Args:
        centres (list): List of center coordinates.

    Returns:
        np.ndarray: A 2D array containing the distances between each pair of centers.
    """
    num_cities = len(centres)
    distances = np.zeros((num_cities, num_cities))
    
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            distances[i, j] = distances[j, i] = "{:.2f}".format(get_euclidean_distance(centres[i], centres[j]) /100)


    return distances

def draw_solution_graph(centres, solution, node_distances, flipped=True):
    """Draw the solution graph.

    Args:
        centres (list): List of center coordinates.
        solution (list): The order of nodes that the rover should visit.
        node_distances (list): The distances between each pair of nodes.
        flipped (bool, optional): Whether to flip the graph vertically. Default is True.
    """

    # Create a graph
    G = nx.Graph()
    
    # Add nodes
    for i in range(len(centres)):
        G.add_node(i, pos=centres[i])
        
    # Add edges
    edges = [(solution[i-1], solution[i]) for i in range(0, len(solution))]
    G.add_edges_from(edges)
    
    pos = nx.get_node_attributes(G, 'pos')

    if flipped:
        pos = {node: (x,-y) for (node, (x,y)) in pos.items()}

    # Draw nodes with light blue color
    nx.draw(G, pos, with_labels=True, node_color='lightblue')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='r', width=2)

    # Add edge labels for distances
    edge_labels = {(solution[i-1], solution[i]): round(node_distances[i], 2) for i in range(len(solution))}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='blue')

    plt.show()


def start_at_zero(solution, node_distances, edge_angles):
    """Start the solution at node 0.'

    Args:
        solution (list): The order of nodes that the rover should visit.
        node_distances (list): The distances between each pair of nodes.
        edge_angles (list): The angles of the edges between each pair of nodes.

    Returns:
        tuple: A tuple containing the rearranged solution, node_distances, and edge_angles.
    """

    # Find the index of node 0
    zero_index = solution.index(0)
    
    # Rearrange the solution to start at node 0
    rearranged_solution = solution[zero_index:] + solution[:zero_index]
    
    # Rearrange node_distances and edge_angles arrays
    rearranged_node_distances = node_distances[zero_index:] + node_distances[:zero_index]
    rearranged_node_distances = rearranged_node_distances[1:] + [rearranged_node_distances[0]]
    
    rearranged_edge_angles = edge_angles[zero_index:] + edge_angles[:zero_index]
    rearranged_edge_angles = rearranged_edge_angles[1:] + [rearranged_edge_angles[0]]

    return rearranged_solution, rearranged_node_distances, rearranged_edge_angles


