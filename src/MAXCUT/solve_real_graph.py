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


path ="/home/gregz/Files/simula_proj/test_images/01.jpg"

G = img_to_graph(path, print_image=True)
draw_graph(G)

w = np.zeros([4,4]) 
for i in range(4): 
    for j in range(4): 
        temp = G.get_edge_data(i, j, default=0)
        if temp != 0: 
            w[i, j] = temp["weight"]
print(w)
