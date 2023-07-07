import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("circles.png")
# img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
t_lower = 50
t_upper = 150

edge = cv.Canny(img, t_lower, t_upper)
cv.imwrite("circlesEdge.png", edge)

circ_gray = cv.imread("circlesEdge.png", cv.IMREAD_UNCHANGED)

_, thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY)
rect = cv.getStructuringElement(cv.MORPH_RECT, (4, 4))
dilation = cv.dilate(thresh, rect, iterations=7)
erosion = cv.erode(dilation, rect, iterations=7)

cv.imshow("", erosion)
cv.waitKey(0)
cv.destroyAllWindows()
