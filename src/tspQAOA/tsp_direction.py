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
