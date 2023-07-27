# PC

import cv2
import tempfile
import socket
import json
import sys
import os

from img_processing.img_to_graph import *
from img_processing.img_tools import *
from path_generator.path_tools import *

import os
import cv2
import socket
import json
import rover  # Assuming this module is imported

def capture_and_send_instructions():
    """Captures an image from the webcam, gets instructions based on the image, 
    and sends those instructions to the rover.
    """
    
    path = os.getcwd()
    sys.path.append(f'{path}/src/')
    os.getcwd()

    sys.path.insert(1, os.path.join(sys.path[0], '../src'))
    
    # Set the path for the temporary image file
    temp_file_path = f'{path}/test_images/temp.jpg'
    
    # Capture image from webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()

        cv2.imshow('Webcam', frame)

        # Check for space key press
        key = cv2.waitKey(1)
        if key == ord(' '):  # Space key
            # Save the image to the temporary file
            cv2.imwrite(temp_file_path, frame)
            cv2.destroyAllWindows()  # Close the window
            break

    cap.release()

    if not ret:
        print("Failed to capture image")
        return

    # Get instructions based on the image
    #instructions = get_instructions(temp_file_path)
    instructions = get_instructions('/home/frida/git/simula_proj/test_images/n04.jpg')

    # Set up client socket
    HOST = '192.168.50.108'  # The remote host (the rover's actual IP address)
    PORT = 50007  # The same port as used by the server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(json.dumps(instructions).encode())
    
    print(f"Instructions sent: {instructions}")

    # Delete the temporary image file
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)


def get_instructions(image_file: str, spatial=False, show_graph=False, show_solution=True) -> str:
    """Gets instructions based on an image.
    
    Args:
        image_file (str): The path to the image file.
        spatial (bool, optional): Whether to use the spatial algorithm or not. Default is False.
        show_graph (bool, optional): Whether to show the graph or not. Default is False.
        
    Returns:
        str: A JSON-encoded list of instructions.
    """

    # Step 1: process image and get the corresponding graph
    if spatial:
        G, centres, edge_lengths = img_to_graph_spatial(image_file, print_image=False)
    else: 
        G, centres, edge_lengths = img_to_graph(image_file, return_edge_lengths=True, print_image=False)

    if show_graph:
        draw_graph(G)

    #------------------- CHANGE WITH SOLUTION IMPLEMENTATION -------------------#

    # Step 2: run algorithm to fing path between the nodes
    solution = create_random_array(len(G)) # return just a random path for now

    #---------------------------------------------------------------------------#

    # step 3: get the directions from the path
    if spatial:
        solution_sorted, total_distance, node_distances, edge_angles = graph_from_solution_spatial(G, solution, draw_graph=True)
    else:
        solution_sorted, total_distance, node_distances, edge_angles = graph_from_solution(G, solution, draw_graph=True)

    if show_solution:
        print(solution_sorted)

    # step 4: give directions to the rover
    directions = give_directions(solution_sorted, edge_angles, node_distances)

    # step 5: return the directions as a JSON-encoded string
    return json.dumps(directions)
