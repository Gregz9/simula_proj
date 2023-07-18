import math
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def get_euclidean_distance(point1, point2):
    """Get Euclidean distance between two points.

    Args:
        point1 (tuple): Point 1 with x and y coordinates.
        point2 (tuple): Point 2 with x and y coordinates.

    Returns:
        float: Euclidean distance between point 1 and point 2.
    """
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)


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


def create_random_array(n):
    """Create a random array of size n.

    Args:
        n (int): Size of the array.

    Returns:
        list: A random array of size n.
    """
    return random.sample(range(n), n)


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
    edges = [(solution[i-1], solution[i]) for i in range(1, len(solution))] + [(solution[-1], solution[0])]
    G.add_edges_from(edges)
    
    pos = nx.get_node_attributes(G, 'pos')

    if flipped:
        pos = {node: (x,-y) for (node, (x,y)) in pos.items()}

    # Draw nodes with light blue color
    nx.draw(G, pos, with_labels=True, node_color='lightblue')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='r', width=2)

    # Add edge labels for distances
    edge_labels = {(solution[i-1], solution[i]): round(node_distances[i-1], 2) for i in range(1, len(solution))} # We start from 1 to match the indices with node_distances
    edge_labels[(solution[-1], solution[0])] = round(node_distances[-1], 2)
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


def graph_from_solution(G: nx.Graph, solution: list, draw_graph: bool=False, start_zero=True) -> tuple:
    """Create a graph from the solution.
    
    Args:   
        G (nx.Graph): The input graph.
        solution (list): The order of nodes that the rover should visit.
        draw_graph (bool, optional): Whether to draw the graph. Default is False.
        start_zero (bool, optional): Whether to start the solution at node 0. Default is True.
        
    Returns:   
        tuple: A tuple containing the total distance, node_distances, and edge_angles. 
    """

    # Get the node positions from the graph
    centres = nx.get_node_attributes(G, 'pos')

    # Calculate the total distance, distances between nodes, and the angles of the edges
    total_distance = 0
    node_distances = []
    edge_angles = []

    for i in range(len(solution)):
        if i == 0:  # The first distance is between the last and first nodes in the solution
            node_distance = G[solution[-1]][solution[0]]['weight']
        else:  # The distance is between the current and previous nodes in the solution
            node_distance = G[solution[i-1]][solution[i]]['weight']

        total_distance += node_distance
        node_distance = round(node_distance, 2)
        node_distances.append(node_distance)

        # Calculate angle of the edge
        edge_angle = calculate_angle(centres[solution[i-1]], centres[solution[i]])
        edge_angle = round(edge_angle, 2)
        edge_angles.append(edge_angle)

    total_distance = round(total_distance, 2)

    if start_zero:
        solution, node_distances, edge_angles = start_at_zero(solution, node_distances, edge_angles)

    if draw_graph:
        draw_solution_graph(centres, solution, node_distances)

    return solution, total_distance, node_distances, edge_angles



def give_directions(solution: list, edge_angles: list, node_distances: list, current_angle=90, start_distance=8, verbose=True) -> list:
    """Give directions to a rover to traverse through the graph.

    Args:
        solution (list): The order of nodes that the rover should visit.
        edge_angles (list): The angles of the edges between each pair of nodes.
        node_distances (list): The distances between each pair of nodes.
        current_angle (float, optional): The current angle of the rover with respect to the x-axis. Defaults to 90.
        start_distance (float, optional): The initial distance of the rover from node 0. Defaults to 8.
        verbose (bool, optional): Whether to print the instructions. Defaults to True.

    Returns:
        list: A list of directions for the rover.
    """
    directions = []

    # Move to the initial node from the starting position
    initial_move_angle = current_angle - 90
    initial_move_distance = start_distance
    directions.append({'action': 'Turn', 'value': initial_move_angle})
    directions.append({'action': 'Move', 'value': round(initial_move_distance, 2)})

    for i in range(len(solution)-1):
        next_angle = edge_angles[i]
        turning_angle = next_angle - current_angle
        
        # Turn to the correct angle before moving to the next node
        directions.append({'action': 'Turn', 'value': round(turning_angle, 2)})

        # Since the rover is currently at the node, it should move to the next node
        directions.append({'action': 'Move', 'value': round(node_distances[i], 2)})

        # The current angle is updated to the next_angle for the next iteration
        current_angle = next_angle
        
    # Turn to the angle towards the first node
    last_angle = edge_angles[-1]
    last_turning_angle = last_angle - current_angle
    directions.append({'action': 'Turn', 'value': round(last_turning_angle, 2)})

    # Move to the first node from the last node
    last_node_distance = node_distances[-1]  # distance from the last node to the first node
    directions.append({'action': 'Move', 'value': round(last_node_distance, 2)})

    if verbose:
        for direction in directions:
            if direction['action'] == 'Move':
                print(f"{direction['action']} {direction['value']:.2f} cm")
            else:
                print(f"{direction['action']} {direction['value']:.2f} degrees")

    return directions