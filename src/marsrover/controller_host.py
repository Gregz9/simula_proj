import socket
import sys
import os
import cv2 as cv
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import json
import argparse

src_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(src_dir)

from qiskit_aer import Aer
from qiskit.tools.visualization import plot_histogram
from qiskit.circuit.library import TwoLocal, EfficientSU2
from qiskit_optimization.applications import Maxcut
from qiskit.algorithms.minimum_eigensolvers import (
    SamplingVQE,
    NumPyMinimumEigensolver,
    QAOA,
)
from qiskit.algorithms.optimizers import SPSA, COBYLA, NFT
from qiskit.algorithms import VQE
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit.primitives import Sampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.visualization import circuit_drawer
from qiskit.circuit import QuantumCircuit

from img_processing.img_to_graph import *
from img_processing.img_tools import *
from path_generator.path_tools import *

argsParse = argparse.ArgumentParser()
argsParse.add_argument("-p", "--port", required=True, type=int, help="Port number to connect sockets over")

args = argsParse.parse_args()

# Connecting to the camera
ports = [0, 1, 2]
try:
    for port in ports:
        camera = cv.VideoCapture(port)
        if not camera.isOpened() and port >= 3:
            raise Exception("Failed to find a port that can be used")
        if not camera.isOpened() and port < 3:
            continue
        if camera.isOpened():
            print(f"Camera connected on port: {port}")
            break
except Exception as E:
    print("No port found! Make sure the camera is connected!")

camera.set(cv.CAP_PROP_FPS, 30.0)
camera.set(cv.CAP_PROP_FOURCC, cv.VideoWriter.fourcc("m", "j", "p", "g"))
camera.set(cv.CAP_PROP_FOURCC, cv.VideoWriter.fourcc("M", "J", "P", "G"))
camera.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

result, image = camera.read()

if result:
    cv.imwrite("graph.jpg", image)
else:
    print("ERROR! NO IMAGE CAPTURED")
    # exit()


# original_image = cv.imread("/home/gregz/Files/simula_proj/test_images/n10.jpg")
# graph, cntr_coords = img_to_graph(
    # path="/home/gregz/Files/simula_proj/test_images/n12.jpg", print_image=False
# )
img_path = src_dir + "/img_processing/final_graph/graph.jpg"
graph, cntr_coords = img_to_graph(img_path, print_image=True, dist_meas=True)
# draw_graph(graph)

N = list(graph.edges())
# X = int(len(N) / 1.3)
X = int(len(N) / 2)

edges_to_remove = random.sample(N, X)
graph.remove_edges_from(edges_to_remove)
draw_graph(graph)
N = len(cntr_coords)
w = np.zeros([N, N])
# true_weights = np.zeros([N, N])
for i in range(N):
    for j in range(N):
        temp = graph.get_edge_data(i, j, default=0)
        if temp != 0:
            w[i, j] = 1  # temp["weight"]
            # true_weights[i, j] = temp["weight"]
# true_weights = np.where(true_weights == 0.0, np.nan, true_weights)

best_cost_brute = 0
for b in range(2**N):
    x = [int(t) for t in reversed(list(bin(b)[2:].zfill(N)))]
    cost = 0
    for i in range(N):
        for j in range(N):
            cost = cost + w[i, j] * x[i] * (1 - x[j])
    if best_cost_brute < cost:
        best_cost_brute = cost
        xbest_brute = x
    # print("case = " + str(x) + " cost = " + str(cost))


brute_solution_colors = ["r" if xbest_brute[i] == 0 else "c" for i in range(N)]
# draw_graph(graph, flipped=True, solution=True, colors=brute_solution_colors)
print("\nBest brute force solution = " + str(xbest_brute))
print("Cost found using brute force = " + str(best_cost_brute))
print("--------------------------------------------------------------")


# Transforming the problem into a max-cut instance
Max_Cut = Maxcut(w)
qp = Max_Cut.to_quadratic_program()
qubitOp, offset = qp.to_ising()
# print("Offset: ", offset)
# print(str(qubitOp))
exact_solution = NumPyMinimumEigensolver()
exact_result = exact_solution.compute_minimum_eigenvalue(qubitOp)


x = Max_Cut.sample_most_likely(exact_result.eigenstate)
print("Exact solution of the Ising Hamiltonian representing MAXCUT")
print("energy:", exact_result.eigenvalue.real)
print("max-cut objective:", abs(exact_result.eigenvalue.real + offset))
print("solution:", x)
print("solution objective:", qp.objective.evaluate(x))
print("--------------------------------------------------------------")

# Variational approach to solving the problem

optimizer = COBYLA(maxiter=10000)
init_ansatz = EfficientSU2(
    qubitOp.num_qubits,
    ["rx", "cy"],
    entanglement="circular",
    reps=1,
    # skip_final_rotation_layer=True,
)
vqe = SamplingVQE(sampler=Sampler(), ansatz=init_ansatz, optimizer=optimizer)
result = vqe.compute_minimum_eigenvalue(qubitOp)

x = Max_Cut.sample_most_likely(result.eigenstate)
print("VQE Solution of the Ising Hamiltonian")
print("energy:", result.eigenvalue.real)
print("time:", result.optimizer_time)
print("max-cut objective:", -np.abs(result.eigenvalue) + offset)
print("solution:", x)
print("solution objective:", qp.objective.evaluate(x))

# plot results
colors = ["r" if x[i] == 0 else "c" for i in range(N)]
print(colors)
draw_graph(graph, solution=True, colors=colors)
solution = x

img_path = src_dir + "/img_processing/final_graph/graph.jpg"
graph2, cntr_coords = img_to_graph(img_path, dist_meas=True)

# solutions = create_random_array(len(cntr_coords))
path_to_walk = [i for i in range(len(cntr_coords))]

solution_sorted, total_distance, node_distances, edge_angles = graph_from_solution(graph2, path_to_walk, draw_graph=True)
directions = give_directions(solution_sorted, edge_angles, node_distances, current_angle=90, start_distance = 8)
solution = [int(solution[i]) for i in range(len(solution))]

solution_data = {"directions" : directions, "colors" : colors, "node_coords": cntr_coords}

json_directions = json.dumps(solution_data, indent=2)

PORT = args.port
# HOST = "192.168.50.108"
HOST = "172.26.0.104"
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print("Socket connected")

try:
    sock.bind((HOST, PORT))
except socket.error:
    print("Binding failed")

sock.listen(1)
print("Socket awaiting connection")
(conn, addr) = sock.accept()
print("Connected")

msg_count = 0
while True:
    
    if msg_count == 1: 
        reply = json_directions
    else: 
        reply = str(input("Enter message"))
        if reply == "terminate": 
            conn.close()
            exit() 
    conn.send(reply.encode("utf-8"))
    
    data = conn.recv(1024).decode("utf-8")
    print("Response: ", data)
    msg_count += 1
conn.close()
