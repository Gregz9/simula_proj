from img_tools import draw_graph
from img_to_graph import img_to_graph
import os

# import sys
# sys.path.insert(0, '../src')

# Test paths:

path = "/home/gregz/Files/simula_proj/test_images/05.jpg"
# path = "test_images/02.jpg"
# path = "test_images/03.jpg"
# path = "test_images/04.jpg"
# path = "test_images/05.jpg"

#get the graph
G, _ = img_to_graph(path, print_image=True)
print(_)
draw_graph(G)

