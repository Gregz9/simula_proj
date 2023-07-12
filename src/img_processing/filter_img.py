import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from math import atan2, cos, sin, sqrt, pi

def file_size(path):
    file_size = os.path.getsize(path)
    # file size in MB
    return round(file_size/1024**2, 2)

def drawAxis(img, p_, q_, color, scale):
  p = list(p_)
  q = list(q_)
 
  ## [visualization1]
  angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
  hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
 
  # Here we lengthen the arrow by a factor of scale
  q[0] = p[0] - scale * hypotenuse * cos(angle)
  q[1] = p[1] - scale * hypotenuse * sin(angle)
  cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)
 
  # create the arrow hooks
  p[0] = q[0] + 9 * cos(angle + pi / 4)
  p[1] = q[1] + 9 * sin(angle + pi / 4)
  cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)
 
  p[0] = q[0] + 9 * cos(angle - pi / 4)
  p[1] = q[1] + 9 * sin(angle - pi / 4)
  cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)
  ## [visualization1]
 
def getOrientation(pts, img):
  ## [pca]
  # Construct a buffer used by the pca analysis
  sz = len(pts)
  data_pts = np.empty((sz, 2), dtype=np.float64)
  for i in range(data_pts.shape[0]):
    data_pts[i,0] = pts[i,0,0]
    data_pts[i,1] = pts[i,0,1]
 
  # Perform PCA analysis
  mean = np.empty((0))
  mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)
 
  # Store the center of the object
  cntr = (int(mean[0,0]), int(mean[0,1]))
  ## [pca]
 
  ## [visualization]
  # Draw the principal components
  cv.circle(img, cntr, 3, (255, 0, 255), 2)
  p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
  p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
  drawAxis(img, cntr, p1, (0, 255, 0), 1)
  drawAxis(img, cntr, p2, (255, 255, 0), 5)

  angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
  ## [visualization]
  # Label with the rotation angle
  # label = "  Rotation Angle: " + str(-int(np.rad2deg(angle)) + 90) + " degrees"
  # textbox = cv.rectangle(img, (cntr[0], cntr[1]-25), (cntr[0] + 250, cntr[1] + 10), (255,255,255), -1)
  # cv.putText(img, label, (cntr[0], cntr[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)
 
  return angle, cntr, p2


# image = cv.imread(args["image"])
# print("Image compressed successfully")
# print(file_size("compressed_image.jpg"))
path = "src/img_processing/IMG_8615.jpg"

# print(file_size(path))
ap = argparse.ArgumentParser()
ap.add_argument("-r", "--resizefactor", required=True, help = "Downscaling factor, values 0.0 to 1.0")
ap.add_argument("-c","--compression", required=True, help = "Compression level")
ap.add_argument("-ks","--kernelsize", required=False, help = "Kernel size for Gaussian blur filtering")
ap.add_argument("-sf","--sobelfilter", required=False, action="store_true", help = "If yes, the processed image will be filtered with sobel kernels")
args = vars(ap.parse_args())
# ap.add_argument("-i","--image", required = True, help = "Path to input file")
try: 
    resize_factor = float(args["resizefactor"])
    if resize_factor < 0.0 or resize_factor > 1.0:
        raise ValueError
except ValueError as e: 
    print("Invalid resize value. (Value has to be float ranging 0.0 to 1.0)")
    exit()

try: 
    compression_level = int(args["compression"])  # Convert the compression level to an integer
    if compression_level < 0 or compression_level > 10: 
        raise ValueError
except: 
    print("Invalid value for compression lvl. (Value has to be integer ranging from 0 to 10)")
    exit()

kernel_size = int(args["kernelsize"]) if args["kernelsize"] else 5


img = cv.imread(path)
width = int(img.shape[1] * resize_factor)
height = int(img.shape[1] * resize_factor)

resized = cv.resize(img, (height, width), interpolation=cv.INTER_AREA)
cv.imwrite("compressed_image.jpg", resized, [cv.IMWRITE_JPEG_QUALITY, compression_level])

if args["kernelsize"]: 
    comp_img = cv.imread("compressed_image.jpg")
    filter_img = cv.GaussianBlur(comp_img, (kernel_size, kernel_size),cv.BORDER_DEFAULT)
else: 
    filter_img = cv.imread("compressed_image.jpg")


cv.imwrite("processed_image.jpg", filter_img)

if args["sobelfilter"]: 
    sobelx = cv.Sobel(filter_img, cv.CV_64F,1,0,ksize=5)
    sobely = cv.Sobel(filter_img, cv.CV_64F,0,1,ksize=5)
    plt.imshow(sobelx)
    plt.show()
else:
    plt.imshow(filter_img)
    plt.show()

# hsv = cv.cvtColor(filter_img, cv.COLOR_BGR2HSV)
gray_filter_img = cv.cvtColor(filter_img, cv.COLOR_BGR2GRAY)
_, bw = cv.threshold(gray_filter_img, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)


# Drawing the global axes in the image with labels
cv.line(filter_img, (0, filter_img.shape[0] // 2), (filter_img.shape[1], filter_img.shape[0] // 2), (0, 255, 0), 2)
cv.putText(filter_img, 'X (Global)', (filter_img.shape[1] - 60, filter_img.shape[0] // 2 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
cv.line(filter_img, (filter_img.shape[1] // 2, 0), (filter_img.shape[1] // 2, filter_img.shape[0]), (0, 0, 255), 2)
cv.putText(filter_img, 'Y (Global)', (filter_img.shape[1] // 2 + 10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


for i, c in enumerate(contours): 
    area = cv.contourArea(c)

    if area < 3700 or 100000 < area: 
        continue 

    cv.drawContours(filter_img, contours, i, (0,0,255), 2)

    angle, cntr_local_axes, p2 = getOrientation(c, filter_img)

cv.imshow('Output image', filter_img)
cv.waitKey(0)
cv.destroyAllWindows() 

cv.imwrite("output_img.jpg", filter_img) 








