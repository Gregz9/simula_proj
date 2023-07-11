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
# args = vars(ap.parse_args())
#
#
# image = cv.imread(args["image"])
# print("Image compressed successfully")
# print(file_size("compressed_image.jpg"))


path = "/home/gregz/Files/simula_proj/src/img_processing/IMG_8613.jpg"

# print(file_size(path))
ap = argparse.ArgumentParser()
ap.add_argument("-r", "--resizefactor", required=True, help = "Downscaling factor, values 0.0 to 1.0")
ap.add_argument("-c","--compression", required=True, help = "Compression level")
ap.add_argument("-ks","--kernelsize", required=False, help = "Kernel size for Gaussian blur filtering")
ap.add_argument("-sf","--sobelfilter", required=False, help = "If yes, the processed image will be filtered with sobel kernels")
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

comp_img = cv.imread("compressed_image.jpg")
filter_img = cv.GaussianBlur(comp_img, (kernel_size, kernel_size),cv.BORDER_DEFAULT)
# # filter_img = cv.GaussianBlur(filter_img, (13, 13),cv.BORDER_DEFAULT)
#
# hsv = cv.cvtColor(filter_img, cv.COLOR_BGR2HSV)
if str(args["sobelfilter"]) == "y": 
    sobelx = cv.Sobel(filter_img, cv.CV_64F,1,0,ksize=5)
    sobely = cv.Sobel(filter_img, cv.CV_64F,0,1,ksize=5)
    plt.imshow(sobelx)
    plt.show()
else:
    plt.imshow(filter_img)
    plt.show()
#
# # cv.imshow("",sobelx)
#
