import cv2
import numpy as np

# Load the image
image = cv2.imread('images/back_sub_img3.jpg')
image1 = cv2.imread("images/result3.jpg")

# Convert the image to the desired color space (e.g., BGR to RGB)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Define the color of interest
color_of_interest = (255, 0, 0)  # Example: Blue color

# Create a mask to isolate the desired color
color_mask = cv2.inRange(image_rgb, color_of_interest, color_of_interest)

# Find contours of the color region
contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Calculate the centroid of the color region
for contour in contours:
    M = cv2.moments(contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # Find the pixel coordinates of the maximum intensity for the selected color within the object
    indices = np.argwhere(color_mask == 255)
    max_intensity_coord = indices[np.argmax(indices[:, 0])]

    # Calculate the angle or direction from the centroid to the maximum intensity point
    dx = max_intensity_coord[1] - cx
    dy = max_intensity_coord[0] - cy
    angle = np.arctan2(dy, dx)
    angle_degrees = np.degrees(angle)

    # Define triangle vertices based on the calculated angle and centroid
    triangle_side_length = 20  # Example: Length of each side of the triangle
    triangle_vertices = np.array([
        [cx + triangle_side_length * np.cos(angle), cy + triangle_side_length * np.sin(angle)],
        [cx + triangle_side_length * np.cos(angle - (2 / 3) * np.pi), cy + triangle_side_length * np.sin(angle - (2 / 3) * np.pi)],
        [cx + triangle_side_length * np.cos(angle + (2 / 3) * np.pi), cy + triangle_side_length * np.sin(angle + (2 / 3) * np.pi)]
    ], dtype=np.int32)

    # Draw the triangle
    cv2.polylines(image, [triangle_vertices], True, color_of_interest, thickness=2)

# Display the image with the drawn triangle
cv2.imshow('Triangle pointing to distinct color intensity', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

