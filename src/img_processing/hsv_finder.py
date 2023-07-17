import cv2
import numpy as np
from imutils import paths


def nothing(x):
    pass

def hsv_finder(image, temp=False):
    # Load image
    #image = cv2.imread('blue_postit_254p6.png')
    #image = cv2.imread(img)

    # Create a window
    cv2.namedWindow('image')

    if temp:
        temp_hMin = 95
        temp_sMin = 59
        temp_hMax = 104

    else:
        temp_hMin = 0
        temp_sMin = 0
        temp_hMax = 104

    # Create trackbars for color change
    # Hue is from 0-179 for Opencv
    cv2.createTrackbar('HMin', 'image', temp_hMin, 179, nothing)
    cv2.createTrackbar('SMin', 'image', temp_sMin, 255, nothing)
    cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
    cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMax', 'image', 0, 255, nothing)

    # Set default value for Max HSV trackbars
    cv2.setTrackbarPos('HMax', 'image', 179)
    cv2.setTrackbarPos('SMax', 'image', 255)
    cv2.setTrackbarPos('VMax', 'image', 255)

    # Initialize HSV min/max values
    hMin = sMin = vMin = hMax = sMax = vMax = 0
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0

    while(1):
        # Get current positions of all trackbars
        hMin = cv2.getTrackbarPos('HMin', 'image')
        sMin = cv2.getTrackbarPos('SMin', 'image')
        vMin = cv2.getTrackbarPos('VMin', 'image')
        hMax = cv2.getTrackbarPos('HMax', 'image')
        sMax = cv2.getTrackbarPos('SMax', 'image')
        vMax = cv2.getTrackbarPos('VMax', 'image')

        # Set minimum and maximum HSV values to display
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        # Convert to HSV format and color threshold
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(image, image, mask=mask)

        # Print if there is a change in HSV value
        if((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
            print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
            phMin = hMin
            psMin = sMin
            pvMin = vMin
            phMax = hMax
            psMax = sMax
            pvMax = vMax

        # Display result image
        cv2.imshow('image', result)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

#pin = 'pin.jpg'
#hsv_finder(pin)

#img = 'tsp_img_ex/0.jpg'
#img = 'tsp_img_ex/1.jpg'
#img = 'tsp_img_ex/2.jpg'
#img = 'tsp_img_ex/3.jpg'
#img = 'tsp_img_ex/4.jpg'
img = 'tsp_img_ex/5.jpg'

img = cv2.imread(img)
img = cv2.resize(img, (0,0), fx=0.2, fy=0.2)
hsv_finder(img, temp=True)