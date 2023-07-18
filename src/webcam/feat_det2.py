import cv2
import numpy as np

# Read the main image and the template
main_image = cv2.imread('monday_17_cl151710.jpg', 0)
template = cv2.imread('IMG_8793.jpg', 0)

# Apply template matching
result = cv2.matchTemplate(main_image, template, cv2.TM_CCOEFF_NORMED)
_, max_val, _, max_loc = cv2.minMaxLoc(result)

# Get the matched template's position
top_left = max_loc
h, w = template.shape[:2]
bottom_right = (top_left[0] + w, top_left[1] + h)

# Calculate the rotation angle
matched_region = main_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
moments = cv2.moments(matched_region)
angle = 0.5 * np.arctan2(2 * moments['mu11'], moments['mu20'] - moments['mu02']) * 180 / np.pi

# Draw a rectangle around the matched region
cv2.rectangle(main_image, top_left, bottom_right, 255, 2)

# Display the matched region and the rotation angle
cv2.imshow('Matched Region', matched_region)
print(f"Rotation angle: {angle} degrees")

cv2.imshow('Main Image', main_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
