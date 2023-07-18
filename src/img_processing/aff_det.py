import cv2
import numpy as np
import matplotlib.pyplot as plt
# Load the reference image and the current image
reference_image = cv2.imread('normalImg/monday_17_cl151515.jpg', cv2.IMREAD_GRAYSCALE)
current_image = cv2.imread('postItImg/monday_17_cl151340.jpg', cv2.IMREAD_GRAYSCALE)
# current_image = cv2.imread('normalImg/monday_17_cl143745.jpg', cv2.IMREAD_GRAYSCALE)

fig,ax = plt.subplots(1,2)

ax[0].imshow(cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB))
ax[0].set_title("Refernce image")

ax[1].imshow(cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB))
ax[1].set_title("Current image")

plt.show()


# Initialize the SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute local descriptors for the reference image
reference_keypoints, reference_descriptors = sift.detectAndCompute(reference_image, None)

# Detect keypoints and compute local descriptors for the current image
current_keypoints, current_descriptors = sift.detectAndCompute(current_image, None)

# Match keypoints between the reference and current images
matcher = cv2.FlannBasedMatcher()
matches = matcher.knnMatch(reference_descriptors, current_descriptors, k=2)

# Apply Lowe's ratio test to filter good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Estimate the transformation between the keypoints of reference and current images
reference_points = [reference_keypoints[m.queryIdx].pt for m in good_matches]
current_points = [current_keypoints[m.trainIdx].pt for m in good_matches]
transformation, _ = cv2.estimateAffine2D(np.array(reference_points), np.array(current_points))

# Extract the rotation angle from the transformation matrix
angle_rad = np.arctan2(transformation[1, 0], transformation[0, 0])
angle_deg = -np.degrees(angle_rad)

# Print the orientation angle
print("Orientation Angle: ", angle_deg)

