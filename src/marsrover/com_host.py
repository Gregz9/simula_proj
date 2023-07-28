import socket
import os
import sys
import numpy as np 
import matplotlib.pyplot as plt 
import json
src_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(src_dir)

from path_generator.path_tools import *
from img_processing.img_tools import *
from img_processing.img_to_graph import * 

img_path = src_dir + "/img_processing/final_graph/graph.jpg"
print(img_path)
img = load_img(img_path)
mask = img_treshold(img, HMin=145, SMin=96, VMin=0, HMax=179, SMax=255, VMax=255)
contours, hierarchy = get_countours(mask, 100)
img = draw_contours(img, contours, hierarchy, color=(255, 0, 0))
centres = get_centre(contours)
img = draw_centre(img, centres, color=(255, 0, 0))

sorted_pts = sort_points(centres)
real_width_cm, real_height_cm = 150.0 - 7.6, 150.0 - 7.6
img = descrew(img, sorted_pts, real_width_cm, real_height_cm, dist_meas=True)
centres = detect_nodes(img)

solution = create_random_array(len(centres))
G, _ = img_to_graph(img_path)
draw_graph(G)
solution_sorted, total_distance, node_distances, edge_angles = graph_from_solution(G, solution, draw_graph=True)

directions = give_directions(solution_sorted, edge_angles, node_distances, current_angle=90, start_distance=8)

# Prepping directions for transfer over socket
json_directions = json.dumps(directions)

# HOST = "192.168.50.108"
HOST = "172.26.0.104"
PORT = 12344
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print("Socket connected")

try:
    s.bind((HOST, PORT))
except socket.error:
    print("Bind failed")

s.listen()
print("Socket awaiting message")
(conn, addr) = s.accept()
print('Connected')


msg_count = 0
while True:

    if msg_count == 1:
        reply = json_directions
    else: 
        reply = str(input("Enter a message: "))

    conn.send(reply.encode("utf-8"))
    if reply == "terminate": 
        conn.close()
        exit()

    response = conn.recv(1024).decode("utf-8")
    print(response)
    msg_count += 1
conn.close()
