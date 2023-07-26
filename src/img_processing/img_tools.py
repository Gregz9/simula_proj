import cv2
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import math
import scipy.spatial


def load_img(path):
    """Loads an image from the specified file path and resizes it by 50%.

    Args:
        path (str): Path of the image file.

    Returns:
        ndarray: Resized image.
    """
    img = cv2.imread(path)
    return cv2.resize(img, (0, 0), fx=0.5, fy=0.5)


def print_img(img):
    """Prints an image using matplotlib.

    Args:
        img (ndarray): Image to be printed.
    """
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


def img_treshold(img, HMin=95, SMin=59, VMin=0, HMax=104, SMax=255, VMax=255):
    """Applies threshold to an image based on HSV values.

    Args:
        img (ndarray): Image on which threshold is to be applied.
        HMin, SMin, VMin, HMax, SMax, VMax (int, optional): Min and Max values for Hue, Saturation and Value.

    Returns:
        ndarray: Thresholded image.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (HMin, SMin, VMin), (HMax, SMax, VMax))
    return mask


def get_countours(img, min_contour_area=500):
    """Get contours from an image.

    Args:
        img (ndarray): Image from which to get contours.
        min_contour_area (int, optional): Minimum contour area to consider. Default is 500.

    Returns:
        list: List of contours which area is greater than the minimum contour area.
        hierarchy: Hierarchy of the contours.
    """
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    valid_contours = []
    valid_hierarchy = []

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > min_contour_area:
            valid_contours.append(cnt)
            valid_hierarchy.append(hierarchy[0][i])

    return valid_contours, np.array([valid_hierarchy])


def draw_contours(img, contours, hierarchy, idx=-1, color=(0, 255, 0), thickness=2):
    """Draws contours on an image.

    Args:
        img (ndarray): Image on which to draw contours.
        contours (list): Contours to be drawn.
        hierarchy (ndarray): Hierarchy of contours.
        idx (int, optional): Parameter indicating a contour to draw. If it is negative, all the contours are drawn.
        color (tuple, optional): Color of the contours.
        thickness (int, optional): Thickness of the contours.

    Returns:
        ndarray: Image with contours drawn.
    """
    cv2.drawContours(img, contours, idx, color, thickness, hierarchy=hierarchy)
    return img


def get_centre(contours):
    """Finds the center of contours.

    Args:
        contours (list): List of contours.

    Returns:
        list: List of center points for all contours.
    """
    centres = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centres.append((cx, cy))
    return centres


def draw_centre(img, centres, color=(0, 0, 255), thickness=2):
    """Draws center points on an image.

    Args:
        img (ndarray): Image on which to draw center points.
        centres (list): List of center points.
        color (tuple, optional): Color of the center points.
        thickness (int, optional): Thickness of the center points.

    Returns:
        ndarray: Image with center points drawn.
    """
    for centre in centres:
        cv2.circle(img, centre, 5, color, thickness)
    return img


def sort_points(centres):
    """Sorts a list of points based on their sum and difference.

    Args:
        centres (list): List of points to be sorted.

    Returns:
        ndarray: Array of sorted points.
    """

    # The top-left point will have the smallest sum whereas
    # the bottom-right point will have the largest sum
    centres = np.asarray(centres)
    s = centres.sum(axis=1)
    diff = np.diff(centres, axis=1)

    # Top-left point has smallest sum...
    # np.argmin(s) gives the index of min element
    # 'centres' is an array so we use centres[index] to access the element
    tl = centres[np.argmin(s)]

    # Bottom-right point has largest sum
    br = centres[np.argmax(s)]

    # Top-right point has smallest difference
    tr = centres[np.argmin(diff)]

    # Bottom-left point has largest difference
    bl = centres[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype="float32")


def descrew(img, sorted_pts, real_width_cm, real_height_cm, dist_meas=False):
    """Corrects the perspective of an image.

    Args:
        img (ndarray): Image to be corrected.
        sorted_pts (ndarray): Sorted points defining the perspective.
        real_width_cm (float): Actual width of the object in the image.
        real_height_cm (float): Actual height of the object in the image.

    Returns:
        ndarray: Image with corrected perspective.
    """

    # Given that you know the real-world distances between the corner points,
    # we can assume the destination image should have this size.
    # We'll convert it to pixels assuming a DPI of 96 (common for web images).
    # This is used only for visualizing the result. For further calculations,
    # you'd need to keep using the real-world distances instead of pixels.

    dpi = 96  # Common DPI for web images, change it if you know the actual DPI of your image.
    dst_width_px = int(real_width_cm * dpi / 2.54)
    dst_height_px = int(real_height_cm * dpi / 2.54)

    if not dist_meas:
        dst = np.array(
            [
                [0, 0],
                [dst_width_px - 1, 0],
                [dst_width_px - 1, dst_height_px - 1],
                [0, dst_height_px - 1],
            ],
            dtype="float32",
        )
    else:
        dst = np.array(
            [
                [0, 0],
                [dst_width_px // 5 - 1, 0],
                [dst_width_px // 5 - 1, dst_height_px // 5 - 1],
                [0, dst_height_px // 5 - 1],
            ],
            dtype="float32",
        )

    # Calculate the perspective transform matrix and warp the perspective.
    M = cv2.getPerspectiveTransform(sorted_pts, dst)
    dst_width_px = dst_width_px // 5 if dist_meas else dst_width_px
    dst_height_px = dst_height_px // 5 if dist_meas else dst_height_px

    warp = cv2.warpPerspective(img, M, (dst_width_px, dst_height_px))
    print(f"New width: {dst_width_px}")
    print(f"New height: {dst_height_px}")

    return warp


def create_graph(centres, img, return_edge_lengths=False):
    """Creates a graph from a list of points.

    Args:
        centres (list): List of points.
        img (ndarray): Image related to the points.
        return_edge_lengths (bool, optional): Whether to return the lengths of the edges. Defaults to False.

    Returns:
        nx.Graph: Graph created from the points.
        list: List of lengths of the edges (if return_edge_lengths is True).
    """

    # Create a new graph
    G = nx.Graph()

    # Calculate real world distance per pixel
    real_width_cm, real_height_cm = 80 - 7.6, 96.5 - 7.6
    px_per_cm_x = img.shape[1] / real_width_cm
    px_per_cm_y = img.shape[0] / real_height_cm

    # Add a node for each centre
    for i, centre in enumerate(centres):
        pos_cm = (
            centre[0] / px_per_cm_x,
            centre[1] / px_per_cm_y,
        )  # Calculate position in cm
        G.add_node(i, pos=pos_cm)

    edge_lengths = []

    # Connect each node to every other node
    for i in range(len(centres)):
        for j in range(i + 1, len(centres)):
            # Calculate weight (distance in cm)
            line_length_px = math.sqrt(
                (centres[j][0] - centres[i][0]) ** 2
                + (centres[j][1] - centres[i][1]) ** 2
            )
            line_length_cm_x = line_length_px / px_per_cm_x
            line_length_cm_y = line_length_px / px_per_cm_y
            # The weight is set to the average of the distances in the x and y directions
            weight = round(((line_length_cm_x + line_length_cm_y) / 2), 2)
            G.add_edge(i, j, weight=weight)
            edge_lengths.append(weight)

    if return_edge_lengths:
        return G, edge_lengths
    else:
        return G


def draw_graph(G, flipped=True, solution=False, colors=["skyblue"]):
    """Draws a graph using matplotlib.

    Args:
        G (nx.Graph): Graph to be drawn.
        flipped (bool, optional): Whether to flip the graph vertically. Default is True.
        solution (bool, optional): Whether to draw the graph as a solution. Default is False.
        colors (list, optional): List of colors to be used for the nodes. Default is ["skyblue"].
    """
    pos = nx.get_node_attributes(G, "pos")
    labels = nx.get_edge_attributes(G, "weight")

    plt.figure(figsize=(12, 12))
    if flipped:
        pos = {node: (x, -y) for (node, (x, y)) in pos.items()}

    node_color = "skyblue" if not solution else colors
    nx.draw(
        G, pos, with_labels=True, node_color=node_color, node_size=1500, font_size=20
    )
    edge_labels_pos = {}

    for edge in G.edges:
        start_pos = np.array(pos[edge[0]])
        end_pos = np.array(pos[edge[1]])
        center_pos = (start_pos + end_pos) / 2
        edge_labels_pos[edge] = center_pos + np.array([0.05, 0.05])

    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    plt.show()


def draw_centre_and_lines(img, centres, color=(0, 0, 255), thickness=2):
    """Draws center points and lines between them on an image.

    Args:
        img (ndarray): Image on which to draw.
        centres (list): List of center points.
        color (tuple, optional): Color of the center points and lines.
        thickness (int, optional): Thickness of the center points and lines.

    Returns:
        ndarray: Image with center points and lines drawn.
    """

    # Calculate real world distance per pixel
    real_width_cm, real_height_cm = 80 - 7.6, 96.5 - 7.6
    px_per_cm_x = img.shape[1] / real_width_cm
    px_per_cm_y = img.shape[0] / real_height_cm

    # Draw centre for all contours and lines between them
    for i, centre in enumerate(centres):
        cv2.circle(img, centre, 5, color, thickness)
        for other_centre in centres[i + 1 :]:
            cv2.line(img, centre, other_centre, color, thickness)
            # Calculate and print length of the line in cm
            line_length_px = math.sqrt(
                (other_centre[0] - centre[0]) ** 2 + (other_centre[1] - centre[1]) ** 2
            )
            line_length_cm_x = line_length_px / px_per_cm_x
            line_length_cm_y = line_length_px / px_per_cm_y
            print(
                f"Length of line between {centre} and {other_centre} in x direction: {line_length_cm_x:.2f} cm"
            )
            print(
                f"Length of line between {centre} and {other_centre} in y direction: {line_length_cm_y:.2f} cm"
            )
    return img


def relative_orientation(template: str, current: str) -> np.float32:
    reference_frame = cv2.imread(template, cv2.IMREAD_GRAYSCALE)
    current_frame = cv2.imread(current, cv2.IMREAD_GRAYSCALE)

    sift = cv2.SIFT_create()

    reference_keypoints, reference_descriptors = sift.detectAndCompute(
        reference_frame, None
    )
    current_keypoints, current_descriptors = sift.detectAndCompute(current_frame, None)

    matcher = cv2.FlannBasedMatcher()
    matches = matcher.knnMatch(reference_descriptors, current_descriptors, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    reference_points = [reference_keypoints[m.queryIdx].pt for m in good_matches]
    current_points = [current_keypoints[m.trainIdx].pt for m in good_matches]
    transformation, _ = cv2.estimateAffine2D(
        np.array(reference_points), np.array(current_points)
    )

    angle_rad = np.arctan2(transformation[1, 0], transformation[0, 0])
    angle_deg = -np.degrees(angle_rad)

    return angle_deg


def real_world_dist(coords_rover, coords_node, ratio_x, ratio_y):
    """
    Computes the real distance between the rover and a node in the graph
    based on the coordinates of the centers of both objects, and ratio
    of distance to pixel. 

    Args: 
        coords_rover (np.ndarray): Two integer values representing the x- and y- coordinates of the rover
        coords_node (np.ndarray): Two integer values representing the x- and y- coordinates of the node
        ratio_x (float): Decimal number representing the ratio of distance per pixel along the x-axis
        ratio_y (float): Decimal number representing the ratio of distance per pixel along the y-axis

    Returns: 
        float representing the real world distance between the rover and one od the nodes in the graph.

    """

    temp = coords_rover - coords_node
    temp[0] *= ratio_y
    temp[1] *= ratio_x
    return np.sqrt(temp[0] ** 2 + temp[1] ** 2)


def detect_rover(img, descrew=True):
    """
    Detects the rover in an image that has not been transformed/"descrewed"

    Args:
        img (np.ndarray): Image to extract coordinates from
        descrew (boolean): Flag which tells if image is transformed or not

    Returns:
        np.ndarray containing the coordinates of the rover in the image

    """
    if not descrew:
        mask = img_treshold(img, HMin=0, SMin=0, VMin=0, HMax=179, SMax=255, VMax=94)
    else:
        mask = img_treshold(img, HMin=0, SMin=0, VMin=0, HMax=179, SMax=255, VMax=101)

    contours, _ = get_countours(mask)
    centres = get_centre(contours)
    # img = draw_centre(img, [centre[0]], color(255,0,0)
    return np.array(centre[0])


def detect_nodes(img, descrew=True):
    """
    Detects the coordinates of nodes in an image that has been "descrewed"

    Args:
        img (np.ndarray): Image to extract coordinates from

    Returns:
        np.ndarray containing the coordinates of the nodes within the image
    """
    mask = img_treshold(
        second_img, HMin=69, SMin=97, Vmin=94, HMax=179, SMax=162, VMax=151
    )
    contours, hierarchy = get_countours(mask, 100)
    centres = get_centre(contours)
    return centres


def create_graph_spatial(centres, img, n_nearest):
    """Creates a graph from a list of points where each point is only connected to its n nearest neighbors.

    Args:
        centres (list): List of points.
        img (ndarray): Image related to the points.
        n_nearest (int): The number of nearest neighbors to connect each node to.

    Returns:
        nx.Graph: Graph created from the points.
        list: List of lengths of the edges.
    """

    # Create a new graph
    G = nx.Graph()

    # Calculate real world distance per pixel
    real_width_cm, real_height_cm = 80 - 7.6, 96.5 - 7.6
    px_per_cm_x = img.shape[1] / real_width_cm
    px_per_cm_y = img.shape[0] / real_height_cm

    # Add a node for each centre
    for i, centre in enumerate(centres):
        pos_cm = (
            centre[0] / px_per_cm_x,
            centre[1] / px_per_cm_y,
        )  # Calculate position in cm
        G.add_node(i, pos=pos_cm)

    edge_lengths = []

    # Build a k-d tree for efficient nearest neighbor search
    tree = scipy.spatial.cKDTree(centres)

    # Connect each node to its n nearest neighbors
    for i, centre in enumerate(centres):
        distances, indices = tree.query(
            centre, k=n_nearest + 1
        )  # Query includes the point itself
        for distance, j in zip(
            distances[1:], indices[1:]
        ):  # Skip the first result (the point itself)
            line_length_px = distance
            line_length_cm_x = line_length_px / px_per_cm_x
            line_length_cm_y = line_length_px / px_per_cm_y
            # The weight is set to the average of the distances in the x and y directions
            weight = round(((line_length_cm_x + line_length_cm_y) / 2), 2)
            G.add_edge(i, j, weight=weight)
            edge_lengths.append(weight)

    return G, edge_lengths
