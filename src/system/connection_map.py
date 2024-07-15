import numpy as np
import cv2
from collections import defaultdict
import math
from torchvision import transforms


from src.neural_network_stuff.custome_dataset import CustomImageDataset
from src.visualize.visualize_image import visualize_image 
from src.visualize.draw_graph import draw_stuff_on_image_and_save
import src.configure as configure


def connection_map(data_set: CustomImageDataset):
    
    # Load Image
    id = data_set.id_list[0]
    img = data_set.image_dic[id]
    data = data_set.label_dic[id]

    # Convert to right datatype
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)


    
    for i, ((x_0, y_0), class_id, degree) in enumerate(data):
        part_img, origin = cut_image_np_safe(img, x_0, y_0) 
        points = check_for_black_pixel_at_border(part_img, origin)

        display_img = draw_stuff_on_image_and_save(img, points, [])
        visualize_image(display_img)



def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def cluster_and_average(points, threshold=5):
    clusters = defaultdict(list)
    
    for point in points:
        added = False
        for center in clusters:
            if distance(point, center) <= threshold:
                clusters[center].append(point)
                added = True
                break
        if not added:
            clusters[point].append(point)
    
    averaged_points = []
    for cluster in clusters.values():
        x_avg = sum(p[0] for p in cluster) / len(cluster)
        y_avg = sum(p[1] for p in cluster) / len(cluster)
        averaged_points.append((round(x_avg), round(y_avg)))
    
    return averaged_points


def check_for_black_pixel_at_border(img_part: np.ndarray, origin: tuple):
    """
    Checks for black pixels on the borders of the given image part and returns their coordinates.

    :param img_part: NumPy array representing the image part.
    :param origin: Tuple (x, y) representing the global coordinates of the top-left corner of the image part.
    :return: A list of absolute coordinates of black pixels on the borders.
    """
    black_pixels_coords = []
    origin_x, origin_y = origin

    # Define the threshold for black pixels considering the margin
    margin = configure.black_pixel_margin
    black_threshold = np.array([margin, margin, margin])

    
    centered_black_pixel_cords = []
    # Top border
    for x in range(img_part.shape[1]):
        if np.all(img_part[0, x] <= black_threshold):
            black_pixels_coords.append((origin_x + x, origin_y))

    centered_black_pixel_cords.extend(cluster_and_average(black_pixels_coords))
    black_pixels_coords = []

    # Bottom border
    for x in range(img_part.shape[1]):
        if np.all(img_part[-1, x] <= black_threshold):
            black_pixels_coords.append((origin_x + x, origin_y + img_part.shape[0] - 1))
    
    centered_black_pixel_cords.extend(cluster_and_average(black_pixels_coords))
    black_pixels_coords = []

    # Left border
    for y in range(img_part.shape[0]):
        if np.all(img_part[y, 0] <= black_threshold):
            black_pixels_coords.append((origin_x, origin_y + y))
    
    centered_black_pixel_cords.extend(cluster_and_average(black_pixels_coords))
    black_pixels_coords = []

    # Right border
    for y in range(img_part.shape[0]):
        if np.all(img_part[y, -1] <= black_threshold):
            black_pixels_coords.append((origin_x + img_part.shape[1] - 1, origin_y + y))
    
    centered_black_pixel_cords.extend(cluster_and_average(black_pixels_coords))
    black_pixels_coords = []

    return centered_black_pixel_cords



def cut_image_np_safe(image: np.ndarray, x: int, y: int):
    cut_out = configure.img_cut_out
    height, width = image.shape[:2]

    # Calculate the bounding box
    start_x = max(x - cut_out, 0)
    end_x = min(x + cut_out, width)
    start_y = max(y - cut_out, 0)
    end_y = min(y + cut_out, height)

    return image[start_y:end_y, start_x:end_x], (start_x, start_y)
