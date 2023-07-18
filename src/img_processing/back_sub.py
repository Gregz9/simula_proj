import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img0 = cv.imread("images/monday_17_1.jpg") 
img1 = cv.imread("images/monday_17_0.jpg")

print(img0.shape)
# print(img1.shape)
img0 = np.asarray(img0)
img1 = np.asarray(img1)

img2 = img1 - img0
img2 = np.where(img2 > 120, img2, 0)
img2 = np.where(img2 < 245, img2, 0)
cv.imwrite("back_sub_img3.jpg", img2)
plt.imshow(img2)
plt.show()
