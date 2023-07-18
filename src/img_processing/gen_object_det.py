import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

# Opening image
img = cv.imread("images/back_sub_monday.jpg")
img1 = cv.imread("images/monday_17_0.jpg")

# OpenCV opens images as BRG
# but we want it as RGB We'll
# also need a grayscale version
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

_, bw = cv.threshold(img_gray, 50, 255, cv.THRESH_BINARY)
kernel = np.ones((5,5), np.uint8)
eroded_img = cv.dilate(bw, kernel, iterations = 1)
kernel = np.ones((10,10), np.uint8)
closed_eroded_img = cv.morphologyEx(eroded_img, cv.MORPH_OPEN, kernel)

eroded_img2 = cv.dilate(closed_eroded_img, kernel, iterations = 1)
plt.imshow(eroded_img2)
plt.show()
contours, _ = cv.findContours(eroded_img2, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

for i, c in enumerate(contours): 
   
    # x,y,w,h = cv.boundingRect(contours[i])
    # cv.rectangle(img, (x,y), (x + w, y + h), (0, 255, 0), 2)
    rect = cv.minAreaRect(contours[i])
    box = cv.boxPoints(rect)
    box = np.int0(box)
    cv.fillPoly(img1, [box], (255,255,255))
    
    angle = rect[-1]
    angle_rad = np.radians(angle)

    # Calculate the orientation angle with respect to the global x-axis
    orientation_angle = np.degrees(np.arctan2(np.cos(angle_rad), np.sin(angle_rad)))

    # Print the orientation angle with respect to the global x-axis
    print("Orientation angle with respect to global x-axis (degrees):", orientation_angle)
    # cv.drawContours(img1, [box], i, (0, 255, 0), 2)
    

# img2 = cv.imread("images/result3.jpg")
plt.imshow(img1)
plt.show()
cv.imwrite("detect_img3.jpg", img1)

