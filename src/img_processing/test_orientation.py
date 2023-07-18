import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt
from img_tools import relative_orientation

template = "/home/gregz/Files/simula_proj/src/webcam/normalImg/monday_17_cl151515.jpg"
current = "/home/gregz/Files/simula_proj/src/webcam/normalImg/monday_17_cl151720.jpg"



angle = relative_orientation(template, current)
print(f"The relative angle difference between two frames: {angle}")

template = cv.imread(template)
current = cv.imread(current)
fig, ax = plt.subplots(1,2) 

ax[0].imshow(cv.cvtColor(template, cv.COLOR_BGR2RGB))
ax[0].set_title("Template/Reference")

ax[1].imshow(cv.cvtColor(current, cv.COLOR_BGR2RGB))
ax[1].set_title("Current Frame")
plt.show()
