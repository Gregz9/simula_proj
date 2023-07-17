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


def graph_from_solution(solution: list, centres: list, draw_graph: bool=False, start_zero=True) -> tuple:

    centres_len = len(centres)
    cost_matrix = np.zeros((centres_len, centres_len))

    for i in range(centres_len):
        for j in range(i + 1, centres_len):
            print(i, j)
            cost_matrix[i, j] = cost_matrix[j, i] = get_euclidean_distance(centres[i], centres[j])

    # Calculate the total distance, distances between nodes, and the angles of the edges
    total_distance = 0
    node_distances = []
    edge_angles = []
    for i in range(centres_len):
        node_distance = cost_matrix[solution[i-1], solution[i]]
        total_distance += node_distance
        node_distances.append(node_distance)

        # Calculate angle of the edge
        edge_angle = calculate_angle(centres[solution[i-1]], centres[solution[i]])
        edge_angle = round(edge_angle, 2)
        edge_angles.append(edge_angle)

    total_distance = round(total_distance, 2)

    if draw_graph:
        draw_solution_graph(centres, solution, node_distances)

    if start_zero:
        solution, node_distances, edge_angles = start_at_zero(solution, node_distances, edge_angles)

    return total_distance, node_distances, edge_angles


def give_directions(solution: list, edge_angles: list, node_distances: list, current_angle=90, start_distance=8) -> list:
    """Give directions to a rover to traverse through the graph.

    Args:
        solution (list): The order of nodes that the rover should visit.
        edge_angles (list): The angles of the edges between each pair of nodes.
        node_distances (list): The distances between each pair of nodes.
        current_angle (float, optional): The current angle of the rover with respect to the x-axis. Defaults to 90.
        start_distance (float, optional): The initial distance of the rover from node 0. Defaults to 8.

    Returns:
        list: A list of directions for the rover.
    """
    directions = []

    # Move to the initial node from the starting position
    initial_move_distance = start_distance
    directions.append(f'Move {initial_move_distance:.2f} cm')

    for i in range(len(solution) - 1):
        next_angle = edge_angles[i]
        turning_angle = next_angle - current_angle
        turning_direction = 'right' if turning_angle < 0 else 'left'
        
        # Turn to the correct angle before moving to the next node
        turn_angle = f'Turn {abs(turning_angle):.2f} degrees {turning_direction}'
        directions.append(turn_angle)

        # Since the rover is currently at the node, it should move to the next node
        move_distance = f'Move {node_distances[i]:.2f} cm'
        directions.append(move_distance)

        # The current angle is updated to the next_angle for the next iteration
        current_angle = next_angle
        
    # Return to the first node from the last node in the solution
    last_node_distance = node_distances[0]  # distance from the last node to the first node
    last_node_move = f'Move {last_node_distance:.2f} cm'
    directions.append(last_node_move)

    return directions