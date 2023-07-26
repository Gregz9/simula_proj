import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from img_tools import *


def real_world_dist(coords_rover, coords_node, ratio_x, ratio_y):
    temp = coords_rover - coords_node
    temp[0] *= ratio_y
    temp[1] *= ratio_x
    return np.sqrt(temp[0] ** 2 + temp[1] ** 2)

# print(real_world_dist(np.array([0, 0]), np.array([1454, 1266]), 142.4/1266, 142.4/1454))

TEMPLATE = "feat_img/feat_det0.jpg"
DEFAULT_ORIENTATION = 90.0

# ---- Detecting the borders ----
first_img = load_img("final_graph/graph_rover4.jpg")
mask = img_treshold(first_img, HMin=145, SMin=96, VMin=0, HMax=179, SMax=255, VMax=255)
contours, hierarchy = get_countours(mask, 100)
img = draw_contours(first_img, contours, hierarchy, color=(255, 0, 0))
centres = get_centre(contours)
img = draw_centre(img, centres, color=(255, 0, 0))

sorted_pts = sort_points(centres)
real_width_cm, real_height_cm = 175.1 - 7.6, 200.0 - 7.6
img = descrew(img, sorted_pts, real_width_cm, real_height_cm, dist_meas=True)

plt.imshow(img)
plt.show()

cv.imwrite("descrewed_graph.jpg", img)

# ---- Orientation of the rover ----
turn_deg = relative_orientation(TEMPLATE, "img_graph/rover_0.jpg")
print("Relative orientation: ", turn_deg)
print("Orientation w.r.t the global axes: ", DEFAULT_ORIENTATION + turn_deg)

# ---- Detecting the location of the nodes ----
second_img = load_img("final_graph/graph.jpg.jpg")
mask2 = img_treshold(second_img, HMin=44, SMin=37, VMin=0, HMax=148, SMax=255, VMax=255)
contours, hierarchy = get_countours(mask2, 100)
img = draw_contours(second_img, contours, hierarchy, color=(0, 255, 0))
centres = get_centre(contours)
print(centres)
img = draw_centre(img, centres, color=(0, 255, 0))
plt.imshow(img)
plt.show()

# ---- Detecting the location of rover ----
third_img = load_img("final_graph/graph_empty_rover3.jpg")
# mask3 = img_treshold(third_img, HMin=0, SMin=0, VMin=0, HMax=179, SMax=255, VMax=100)
# contours, hierarchy = get_countours(mask3, 100)
# print(len(contours))
# img = draw_contours(third_img, contours, hierarchy, (255,0, 0))
# centres = get_centre(contours)
# img = draw_centre(img, [centre[1]], color=(255,0,0))

mask = img_treshold(third_img, HMin=0, SMin=0, VMin=0, HMax=179, SMax=255, VMax=94)

plt.imshow(mask) 
plt.show()

contours, hierarchy = get_countours(mask)
img = cv.drawContours(third_img, contours, 0, (255, 0, 0))

centres = get_centre(contours)
img = draw_centre(img, [centres[0]], color=(255, 0, 0))


plt.imshow(img)
plt.show()
