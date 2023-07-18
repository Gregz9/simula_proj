import cv2
import numpy as np

def estimate_starting_orientation(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection (optional)
    edges = cv2.Canny(gray_image, threshold1=50, threshold2=150)

    # Use Hough Line Transform to detect lines in the image
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=100)

    # Use RANSAC to estimate the dominant line representing the rover's orientation
    best_line = None
    max_inliers = 0
    for line in lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            # Generate two points on the line
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            # Count inliers (points close to the line)
            inliers = np.sum(np.abs(edges - cv2.line(np.zeros_like(edges), (x1, y1), (x2, y2), 255, 1)) < 5)

            if inliers > max_inliers:
                max_inliers = inliers
                best_line = (rho, theta)

    # Calculate the orientation of the dominant line
    if best_line is not None:
        rho, theta = best_line
        starting_orientation = np.degrees(theta)
    else:
        starting_orientation = None

    return starting_orientation

if __name__ == "__main__":
    # Load the image from the camera
    image = cv2.imread('normalImg/monday_17_cl151710.jpg')

    # Estimate the rover's starting orientation
    starting_orientation = estimate_starting_orientation(image)
    print("Rover's Starting Orientation w.r.t. Global X-Axis: ", starting_orientation)


