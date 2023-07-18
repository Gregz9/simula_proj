import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt

from docplex.mp.model import Model 
from qiskit_optimization.translators import from_docplex_mp 
from qiskit.utils import algorithm_globals, QuantumInstance 
from qiskit import Aer
from qiskit.algorithms import QAOA, VQE
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters.quadratic_program_to_qubo import QuadraticProgramToQubo

seed = 1137
n = 4
coordinates = np.random.default_rng(seed).uniform(low=0, high=100, size=(n,2))
pos = dict()
for i, coordinate in enumerate(coordinates): 
    pos[i] = (coordinate[0], coordinate[1])

high = 100 
low = 0 
graph = nx.random_geometric_graph(n=n, radius=np.sqrt((high-low)**2 + (high-low)**2) +1, pos=pos)

for w, v in graph.edges: 
    delta = []
    for i in range(2): 
        delta.append(graph.nodes[w]["pos"][i] - graph.nodes[v]["pos"][i]) 
    graph.edges[w, v]["weight"] = np.rint(np.sqrt(delta[0]**2 + delta[1]**2))

index = dict(zip(list(graph), range(n)))
A = np.full((n,n), np.nan)
for u, wdict in graph.adjacency(): 
    for v, d in wdict.items(): 
        A[index[u], index[v]] = d.get("weight", 1)

A[np.isnan(A)] = 0.0 
A = np.asarray(A) 
M = np.asmatrix(A)

def draw_graph(G, colors, pos): 
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(G, node_color=colors, node_size=600, alpha=.8, ax=default_axes, pos=pos)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels= edge_labels)

colors = ["r" for node in graph.nodes]
pos = [graph.nodes[node]['pos'] for node in graph.nodes]
draw_graph(graph, colors, pos)
plt.show()
mdl = Model(name="TSP")
x = dict() 
for i in range(n): 
    for j in range(n): 
        x[(i, j)] = mdl.binary_var(name="x_{0}_{1}".format(i, j))

C_x = mdl.sum(
        M[i,j] * x[(i,k)] * x[(j, (k + 1) % n)]
        for i in range(n)
        for j in range(n)
        for k in range(n)
        if i != j )

mdl.minimize(C_x)

for i in range(n): 
    mdl.add_constraint(mdl.sum(x[(i, p)] for p in range(n)) == 1) 
for p in range(n): 
    mdl.add_constraint(mdl.sum(x[(i, p)] for i in range(n)) == 1) 

qp = from_docplex_mp(mdl)
qubo = QuadraticProgramToQubo().convert(problem=qp)

def route_x(x): 
    n = int(np.sqrt(len(x)))
    route = []
    for p in range(n): 
        for i in range(n): 
            if x[i * n + p]: 
                route.append(i)
    return route

from qiskit.algorithms.optimizers import COBYLA
import time


optimizer = COBYLA(maxiter=100)
algorithm_globals.random_seed = 10598
backend = Aer.get_backend('qasm_simulator')
backend.set_options(device="GPU")
quantum_instance = QuantumInstance(backend)
qaoa_mes = VQE(quantum_instance = quantum_instance, optimizer=optimizer)
start_time = time.time()
qaoa = MinimumEigenOptimizer(qaoa_mes)
qaoa_result = qaoa.solve(qubo)
print(f"Time taken for VQE solution: {time.time() - start_time}")
print("\nQAOA\n", qaoa_result) 
qaoa_result = np.asarray([int(y) for y in reversed(list(qaoa_result))])
print("\nRoute\n", route_x(qaoa_result))

    



