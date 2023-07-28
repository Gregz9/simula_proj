# useful additional packages
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import time

from qiskit import Aer, QuantumCircuit
from qiskit_aer import Aer
from qiskit.tools.visualization import plot_histogram
from qiskit.circuit.library import TwoLocal, EfficientSU2
from qiskit.primitives import BaseEstimator
from qiskit_optimization.applications import Maxcut, Tsp
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver, QAOA, SamplingVQE
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SPSA
from qiskit.utils import algorithm_globals
from qiskit.primitives import Sampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer


def draw_graph(G, colors, pos):
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(
        G, node_color=colors, node_size=600, alpha=0.8, ax=default_axes, pos=pos
    )
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)


from itertools import permutations


n = 4
num_qubits = n ** 2
tsp = Tsp.create_random_instance(n, seed=98374)
adj_matrix = nx.to_numpy_array(tsp.graph)

print(adj_matrix)

colors = ["r" for node in tsp.graph.nodes]
pos = [tsp.graph.nodes[node]["pos"] for node in tsp.graph.nodes]
print(pos)
draw_graph(tsp.graph, colors, pos)
plt.show()


def brute_force_tsp(w, N):
    a = list(permutations(range(1, N)))
    last_best_distance = 1e10
    for i in a:
        distance = 0
        pre_j = 0
        for j in i:
            distance = distance + w[j, pre_j]
            pre_j = j
        distance = distance + w[pre_j, 0]
        order = (0,) + i
        if distance < last_best_distance:
            best_order = order
            last_best_distance = distance
            print("order = " + str(order) + " Distance = " + str(distance))
    return last_best_distance, best_order


start_time = time.time()
best_distance, best_order = brute_force_tsp(adj_matrix, n)
print(
    "Best order from brute force = "
    + str(best_order)
    + " with total distance = "
    + str(best_distance)
)
print(f"Time taken for brute force solution: {time.time() - start_time}")


def draw_tsp_solution(G, order, colors, pos):
    G2 = nx.DiGraph()
    G2.add_nodes_from(G)
    n = len(order)
    for i in range(n):
        j = (i + 1) % n
        G2.add_edge(order[i], order[j], weight=G[order[i]][order[j]]["weight"])
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(
        G2,
        node_color=colors,
        edge_color="b",
        node_size=600,
        alpha=0.8,
        ax=default_axes,
        pos=pos,
    )
    edge_labels = nx.get_edge_attributes(G2, "weight")
    nx.draw_networkx_edge_labels(G2, pos, font_color="b", edge_labels=edge_labels)


qp = tsp.to_quadratic_program()

from qiskit_optimization.converters import QuadraticProgramToQubo

qp2qubo = QuadraticProgramToQubo()
qubo = qp2qubo.convert(qp)
qubitOp, offset = qubo.to_ising()

exact = MinimumEigenOptimizer(NumPyMinimumEigensolver())
result = exact.solve(qubo)

start_time = time.time()
ee = NumPyMinimumEigensolver()
result = ee.compute_minimum_eigenvalue(qubitOp)
print(f"Time taken for exact solution: {time.time() - start_time}")

print("energy:", result.eigenvalue.real)
print("tsp objective:", result.eigenvalue.real + offset)
x = tsp.sample_most_likely(result.eigenstate)
print("feasible:", qubo.is_feasible(x))
z = tsp.interpret(x)
print("solution:", z)
print("solution objective:", tsp.tsp_value(z, adj_matrix))
draw_tsp_solution(tsp.graph, z, colors, pos)

import sys


from qiskit.algorithms.optimizers import (
    ADAM,
    COBYLA,
    NELDER_MEAD,
    L_BFGS_B,
    POWELL,
    NFT,
    TNC,
)
from qiskit.utils import QuantumInstance

algorithm_globals.random_seed = 73
seed = 10598
sim = Aer.get_backend("aer_simulator")
quantum_instance = QuantumInstance(backend=sim)

optimizer = COBYLA(maxiter=200)
# ry = TwoLocal(qubitOp.num_qubits, "ry", "cz", reps=2, entanglement="linear")
ry = EfficientSU2(qubitOp.num_qubits, ["ry", "cz"], reps=1, entanglement="circular", skip_final_rotation_layer=True)
# vqe = VQE(ansatz=ry, optimizer=optimizer, quantum_instance=quantum_instance)
vqe = SamplingVQE(sampler=Sampler(), ansatz=ry, optimizer=optimizer)
result = vqe.compute_minimum_eigenvalue(qubitOp)

print("energy:", result.eigenvalue.real)
print("time:", result.optimizer_time)
x = tsp.sample_most_likely(result.eigenstate)
print("feasible:", qubo.is_feasible(x))
z = tsp.interpret(x)
print("solution:", z)
print("solution objective:", tsp.tsp_value(z, adj_matrix))
