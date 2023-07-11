import matplotlib.pyplot as plt
import numpy as np 
import networkx as nx

from qiskit_aer import Aer 
from qiskit.tools.visualization import plot_histogram 
from qiskit.circuit.library import TwoLocal 
from qiskit_optimization.applications import Maxcut 
from qiskit.algorithms.minimum_eigensolvers import SamplingVQE, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import SPSA, COBYLA 
from qiskit.utils import algorithm_globals
from qiskit.primitives import Sampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer

from img_tools import draw_graph
from img_to_graph import img_to_graph

path = "/home/gregz/Files/simula_proj/test_images/05.jpg"

G = img_to_graph(path, print_image=True)
draw_graph(G)

w = np.zeros([4,4]) 
for i in range(4): 
    for j in range(4): 
        temp = G.get_edge_data(i, j, default=0)
        if temp != 0: 
            w[i, j] = temp["weight"]

best_cost_brute = 0
for b in range(2**4): 
    x = [int(t) for t in reversed(list(bin(b)[2:].zfill(4)))]
    cost = 0
    for i in range(4): 
        for j in range(4): 
            cost = cost + w[i,j] * x[i] * (1 - x[j])
    if best_cost_brute < cost: 
        best_cost_brute = cost
        xbest_brute = x
    print("case = " + str(x) + " cost = " + str(cost))

colors = ["r" if xbest_brute[i] == 0 else "c" for i in range(4)]
draw_graph(G, flipped=True, solution=True, colors=colors)
print("\nBest solution = " + str(xbest_brute) + "cost = " + str(best_cost_brute))

max_cut = Maxcut(w)
qp = max_cut.to_quadratic_program() 
# print(qp.prettyprint())

qubitOp, offset = qp.to_ising()
# print("Offset: ", offset)
# print("Ising Hamiltonian:")
# print(str(qubitOp))
print("\n")
exact = MinimumEigenOptimizer(NumPyMinimumEigensolver())
result = exact.solve(qp)
# print(result.prettyprint())

ee = NumPyMinimumEigensolver()
result = ee.compute_minimum_eigenvalue(qubitOp)

x = max_cut.sample_most_likely(result.eigenstate)
print("energy:", result.eigenvalue.real)
print("max-cut objective:", abs(result.eigenvalue.real + offset))
print("solution:", x)
print("solution objective:", qp.objective.evaluate(x))

colors = ["r" if xbest_brute[i] == 0 else "c" for i in range(4)]
draw_graph(G, solution=True, colors=colors)
plt.show()

optimizer = COBYLA(maxiter=100)
ry = TwoLocal(qubitOp.num_qubits, "ry", "cz", reps=5, entanglement="linear")
vqe = SamplingVQE(sampler=Sampler(), ansatz=ry, optimizer=optimizer)

# run the VQE algorithm
result = vqe.compute_minimum_eigenvalue(qubitOp)

x = max_cut.sample_most_likely(result.eigenstate)
print("energy:", result.eigenvalue.real)
print("time:", result.optimizer_time)
print("max-cut objective:", result.eigenvalue.real + offset)
print("solution:", x)
print("solution objective:", qp.objective.evaluate(x))

# plot results
colors = ["r" if x[i] == 0 else "c" for i in range(4)]
draw_graph(G, solution=True, colors=colors)
plt.show()
