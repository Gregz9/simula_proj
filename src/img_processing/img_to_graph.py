import networkx as nx
from img_processing.img_tools import *

def img_to_graph(path: str, print_image: bool=False, return_edge_lengths=False, dist_meas=False) -> nx.Graph:
    """Converts an image to a graph.
   
    Args:
       path (str): Path to the image.
       print_img (bool, optional): Whether to print the image or not. Default is False.
       
       Returns:
           nx.Graph: Graph of the image.
    """
    # load image and get contours and centres
        
    img = load_img(path)
    # mask = img_treshold(img, HMin=110, SMin=122, VMin=147, HMax=179)
    mask = img_treshold(img, HMin=145, SMin=96, VMin=0, HMax=179, SMax=255, VMax=255)
    
    #Get the corner nodes for image adjustment
    contours, hierarchy = get_countours(mask, 100)
    img = draw_contours(img, contours, hierarchy, color=(255, 0, 0))
    centres = get_centre(contours)
    img = draw_centre(img, centres, color=(255, 0, 0))

    # descrew image
    sorted_pts = sort_points(centres)
    real_width_cm, real_height_cm = 150.0 - 7.6, 150.0 - 7.6
    img = descrew(img, sorted_pts, real_width_cm, real_height_cm, dist_meas=dist_meas)
    #print_img(img)

    # det the nodes
    mask = img_treshold(img, HMin=49, SMin=65, VMin=0, HMax=104)
    contours, hierarchy = get_countours(mask)
    img = draw_contours(img, contours, hierarchy, color=(0, 0, 255))
    centres = get_centre(contours)
    img = draw_centre(img, centres, color=(0, 0, 255))
    #print_img(img)

    if print_image:
        print_img(img)

    if return_edge_lengths:
        G, edge_lengths = create_graph(centres, img, return_edge_lengths=True)
        return G, centres, edge_lengths
    else:
        G = create_graph(centres, img)
        return G, centres

def pre_process_img(path: str) -> tuple:
    """Processes an image to find its contours and centers.

    Args:
        path (str): The path to the image to be processed.

    Returns:
        img (ndarray): The processed image.
        centres (list): The list of centre points in the image.
        contours (list): The contours found in the image.
        hierarchy (ndarray): The hierarchy of the contours.
    """

    # load image and get contours and centres    
    img = load_img(path)
    
    mask = img_treshold(img, HMin=110, SMin=122, VMin=147, HMax=179)

    #Get the corner nodes for image adjustment
    contours, hierarchy = get_countours(mask)
    img = draw_contours(img, contours, hierarchy, color=(255, 0, 0))
    centres = get_centre(contours)
    img = draw_centre(img, centres, color=(255, 0, 0))

    # descrew image
    sorted_pts = sort_points(centres)
    real_width_cm, real_height_cm = 80 - 7.6, 96.5 - 7.6
    img = descrew(img, sorted_pts, real_width_cm, real_height_cm)

    # det the nodes
    mask = img_treshold(img)
    contours, hierarchy = get_countours(mask)
    img = draw_contours(img, contours, hierarchy, color=(0, 0, 255))
    centres = get_centre(contours)
    img = draw_centre(img, centres, color=(0, 0, 255))

    return img, centres, contours, hierarchy

def img_to_graph_spatial(path: str, n_nearest=3, print_image: bool=False) -> nx.Graph:
    """Converts an image to a graph.
   
    Args:
       path (str): Path to the image.
       print_img (bool, optional): Whether to print the image or not. Default is False.
       
       Returns:
           nx.Graph: Graph of the image.
    """
    # load image and get contours and centres
        
    img = load_img(path)
    mask = img_treshold(img, HMin=110, SMin=122, VMin=147, HMax=179)
    
    #Get the corner nodes for image adjustment
    contours, hierarchy = get_countours(mask)
    img = draw_contours(img, contours, hierarchy, color=(255, 0, 0))
    centres = get_centre(contours)
    img = draw_centre(img, centres, color=(255, 0, 0))
    #print_img(img)

    # descrew image
    sorted_pts = sort_points(centres)
    real_width_cm, real_height_cm = 80 - 7.6, 96.5 - 7.6
    img = descrew(img, sorted_pts, real_width_cm, real_height_cm)
    #print_img(img)

    # det the nodes
    mask = img_treshold(img)
    contours, hierarchy = get_countours(mask)
    img = draw_contours(img, contours, hierarchy, color=(0, 0, 255))
    centres = get_centre(contours)
    img = draw_centre(img, centres, color=(0, 0, 255))
    #print_img(img)

    if print_image:
        print_img(img)

    G, edge_lengths = create_graph_spatial(centres, img, n_nearest)

    return G, centres, edge_lengths
