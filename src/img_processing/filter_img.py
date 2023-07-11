import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def file_size(path):
    file_size = os.path.getsize(path)
    # file size in MB
    return round(file_size/1024**2, 2)
#
# ap = argparse.ArgumentParser()
# ap.add_argument("-i","--image", required = True, help = "Path to input file")
# ap.add_argument("-c","--compression", required=True, help = "Compression level")
# args = vars(ap.parse_args())
#
# image = cv.imread(args["image"])
# compression_level = int(args["compression"])  # Convert the compression level to an integer
# cv.imwrite("compressed_image.jpg", image, [cv.IMWRITE_JPEG_QUALITY, compression_level])
#
# print("Image compressed successfully")
# print(file_size("compressed_image.jpg"))


path = "/home/gregz/Files/simula_proj/src/img_processing/IMG_8613.jpg"
# print(file_size(path))
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--scalefactor", required=True, help = "Downscaling factor, values 0.0 to 1.0")
args = vars(ap.parse_args())
try: 
    scale_factor = float(args["scalefactor"])
    if scale_factor < 0.0 or scale_factor > 1.0:
        raise ValueError
except ValueError as e: 
    print("Invalid compression value")
    exit()


img = cv.imread(path)
width = int(img.shape[1] * scale_factor)
height = int(img.shape[1] * scale_factor)

resized = cv.resize(img, (height, width), interpolation=cv.INTER_AREA)
plt.imshow(resized)
plt.show()
# filter_img = cv.GaussianBlur(img, (21,21),cv.BORDER_DEFAULT)
# # filter_img = cv.GaussianBlur(filter_img, (13, 13),cv.BORDER_DEFAULT)
#
# hsv = cv.cvtColor(filter_img, cv.COLOR_BGR2HSV)
# sobelx = cv.Sobel(hsv, cv.CV_64F,1,0,ksize=5)
# sobely = cv.Sobel(hsv, cv.CV_64F,0,1,ksize=5)
#
# # cv.imshow("",sobelx)
#
# plt.imshow(sobely)
# plt.show()
