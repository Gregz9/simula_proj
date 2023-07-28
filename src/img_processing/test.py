from img_tools import draw_graph
from img_to_graph import img_to_graph
import os
import matplotlib.pyplot as plt
import cv2 as cv
# import sys
# sys.path.insert(0, '../src')

# Test paths:

path1 = "/home/gregz/Files/simula_proj/test_images/05.jpg"
path2 = "/home/gregz/Files/simula_proj/src/img_processing/final_graph/graph.jpg"
fig, ax = plt.subplots(1,2)
ax[0].imshow(cv.imread(path1))
ax[1].imshow(cv.imread(path2))
plt.show()
# path = "test_images/02.jpg"
# path = "test_images/03.jpg"
# path = "test_images/04.jpg"
# path = "test_images/05.jpg"

#get the graph
G, _ = img_to_graph(path2, print_image=True)
print(_)
draw_graph(G)

