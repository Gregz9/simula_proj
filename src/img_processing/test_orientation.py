import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt
from img_tools import relative_orientation

angles = []
# for i in range(2,9,1): 
template = f"feat_img/feat_det0.jpg"
# current = f"feat_img/feat_det2.jpg"
# template = f"normalImg/1.jpg"
# current = f"normalImg/{i}.jpg"
current = f"img_graph/rover_19.jpg"
# current = "descrew_rover.jpg"

angle = relative_orientation(template, current)
angles.append(angle)
print(f"The relative angle difference between two frames: {angle}")

template = cv.imread(template)
current = cv.imread(current)
fig, ax = plt.subplots(1,2) 

ax[0].imshow(cv.cvtColor(template, cv.COLOR_BGR2RGB))
ax[0].set_title("Template/Reference")

ax[1].imshow(cv.cvtColor(current, cv.COLOR_BGR2RGB))
ax[1].set_title("Current Frame")
plt.show()

# print(angles)
