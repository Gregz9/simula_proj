from qiskit_optimization.applications import Tsp
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit import Aer
from qiskit.utils import algorithm_globals
from qiskit.algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_optimization.converters import QuadraticProgramToQubo

import numpy as np
from tspQAOA.tsp_tools import *
from tspQAOA.tsp_direction import *

def tsp_solver(centres: list, distances: list=None, draw_graph: bool=False, rand_seed=1337, start_zero=True) -> list:
    """Try solve the TSP using the QAOA, if fail, use NumPyMinimumEigensolver.

    Args:
        centres (list): List of center coordinates.
        distances (list, optional): List of distances between each pair of centers. Defaults to None.
        draw_graph (bool, optional): Whether to draw the graph. Defaults to False.

    Returns:
        list: A list of directions for the rover.
    """
    
    try:
        solution, total_distance, node_distances, edge_angles = solve_tsp_with_qaoa(centres, distances, draw_graph, rand_seed, start_zero)
    except:
        print('QAOA failed')
        print('Solving with NumPy')
        solution, total_distance, node_distances, edge_angles = solve_tsp_with_numpy(centres, distances, draw_graph, start_zero)

    return solution, total_distance, node_distances, edge_angles


def solve_tsp_with_qaoa(centres: list, distances: list=None, draw_graph: bool=False, rand_seed=1337, start_zero=True) -> list:
    """Solve the TSP using the QAOA.

    Args:
        centres (list): List of center coordinates.
        distances (list, optional): List of distances between each pair of centers. Defaults to None.
        draw_graph (bool, optional): Whether to draw the graph. Defaults to False.

    Returns:
        list: A list of directions for the rover.
    """

    #print(qiskit.__qiskit_version__)

    # Using the centers as the cities, generate a cost matrix based on Euclidean distances or provided distances
    num_cities = len(centres)
    cost_matrix = np.zeros((num_cities, num_cities))

    if distances is None:
        for i in range(num_cities):
            for j in range(i + 1, num_cities):
                cost_matrix[i, j] = cost_matrix[j, i] = get_euclidean_distance(centres[i], centres[j])
    else:
        if len(distances) != num_cities:
            raise ValueError('The length of distances list should be equal to the number of centers')
        cost_matrix = np.array(distances)
        #print(f'Cost matrix: {cost_matrix}')

    # Create an instance of the TSP based on the cost matrix
    tsp = Tsp(cost_matrix)

    # Formulate the problem as a Quadratic Program
    qp = tsp.to_quadratic_program()

    # Convert the problem to a QUBO
    converter = QuadraticProgramToQubo()
    qubo = converter.convert(qp)

    # Initialize the QAOA solver
    algorithm_globals.random_seed = rand_seed
    qaoa_mes = QAOA(quantum_instance=Aer.get_backend('aer_simulator'), initial_point=[0., 0.])

    # Use Minimum Eigen Optimizer with QAOA to solve the problem
    qaoa = MinimumEigenOptimizer(qaoa_mes)

    result = qaoa.solve(qubo)
    #print(f'QAOA result: {result}')

    # Interpret the solution
    solution = tsp.interpret(result)
    #print(f'QAOA solution: {solution}')

    # Calculate the total distance, distances between nodes, and the angles of the edges
    total_distance = 0
    node_distances = []
    edge_angles = []
    for i in range(num_cities):
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

    return solution, total_distance, node_distances, edge_angles




def solve_tsp_with_numpy(centres: list, distances: list=None, draw_graph: bool=False, start_zero=True) -> list:
    """Solve the TSP using the NumPyMinimumEigensolver.

    Args:
        centres (list): List of center coordinates.
        distances (list, optional): List of distances between each pair of centers. Defaults to None.
        draw_graph (bool, optional): Whether to draw the graph. Defaults to False.

    Returns:
        list: A list of directions for the rover.
    """

    # Using the centers as the cities, generate a cost matrix based on Euclidean distances or provided distances
    num_cities = len(centres)
    cost_matrix = np.zeros((num_cities, num_cities))

    if distances is None:
        for i in range(num_cities):
            for j in range(i + 1, num_cities):
                cost_matrix[i, j] = cost_matrix[j, i] = get_euclidean_distance(centres[i], centres[j])
    else:
        if len(distances) != num_cities:
            raise ValueError('The length of distances list should be equal to the number of centers')
        cost_matrix = np.array(distances)

    # Create an instance of the TSP based on the cost matrix
    tsp = Tsp(cost_matrix)

    # Formulate the problem as a Quadratic Program
    qp = tsp.to_quadratic_program()

    # Convert the problem to a QUBO
    converter = QuadraticProgramToQubo()
    qubo = converter.convert(qp)

    # Use Minimum Eigen Optimizer with NumpyMinimumEigensolver to solve the problem
    numpy_solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
    result = numpy_solver.solve(qubo)

    # Interpret the solution
    solution = tsp.interpret(result)

    # Calculate the total distance, distances between nodes, and the angles of the edges
    total_distance = 0
    node_distances = []
    edge_angles = []
    for i in range(num_cities):
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

    return solution, total_distance, node_distances, edge_angles
