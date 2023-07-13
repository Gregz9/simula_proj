import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Tuple, List


def findTurnDegree(obj_coords: Tuple, point_coords: Tuple): 
    
    return -np.rad2deg(np.arctan2((point_coords[1] - obj_coords[1]), (point_coords[0] - obj_coords[0]))) + 90

# coordinates of the tip of vector of objects orientation
obj_coords = (378.27, 284.70)
point_coords = (-640, 1568)

print(findTurnDegree(obj_coords, point_coords))


