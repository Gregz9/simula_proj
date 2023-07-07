import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("circles.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5, 5), cv2.BORDER_DEFAULT)
ret, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY_INV)

cv2.imwrite("thresh.png", thresh)

img_thresh = cv2.imread("thresh.png")

contours, hierarchies = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
blank = np.zeros(thresh.shape[:2], dtype="uint8")
cv2.drawContours(blank, contours, -1, (255, 0, 0), 1)
cv2.imwrite("Contours.png", blank)

for i in contours:
    M = cv2.moments(i)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.drawContours(image, [i], -1, (0, 255, 0), 2)
        cv2.circle(image, (cx, cy), 7, (0, 255, 0), -1)
        cv2.putText(
            image,
            "center",
            (cx - 20, cy - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
        )
    print(f"x: {cx} y: {cy}")

cv2.imwrite("result.png", image)
