import numpy as np
import cv2
from torchvision import transforms
from src.functions import ask_for_dataset, load_datasets
from src.visualize.visualize_image import visualize_image 
from src.visualize.draw_graph import draw_stuff_on_image_and_save
import src.configure as configure

transform_reverse = transforms.Compose([
    transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]),
    transforms.Resize((840, 960)),
])

def cut_image_np_safe(image: np.ndarray, x: int, y: int):
    cut_out = configure.img_cut_out
    height, width = image.shape[:2]

    # Calculate the bounding box
    start_x = max(x - cut_out, 0)
    end_x = min(x + cut_out, width)
    start_y = max(y - cut_out, 0)
    end_y = min(y + cut_out, height)

    return image[start_y:end_y, start_x:end_x], (start_x, start_y)

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

    centered_black_pixel_cords.extend(get_middle_of_black_point_border(black_pixels_coords))
    black_pixels_coords = []

    # Bottom border
    for x in range(img_part.shape[1]):
        if np.all(img_part[-1, x] <= black_threshold):
            black_pixels_coords.append((origin_x + x, origin_y + img_part.shape[0] - 1))
    
    centered_black_pixel_cords.extend(get_middle_of_black_point_border(black_pixels_coords))
    black_pixels_coords = []

    # Left border
    for y in range(img_part.shape[0]):
        if np.all(img_part[y, 0] <= black_threshold):
            black_pixels_coords.append((origin_x, origin_y + y))
    
    centered_black_pixel_cords.extend(get_middle_of_black_point_border(black_pixels_coords))
    black_pixels_coords = []

    # Right border
    for y in range(img_part.shape[0]):
        if np.all(img_part[y, -1] <= black_threshold):
            black_pixels_coords.append((origin_x + img_part.shape[1] - 1, origin_y + y))
    
    centered_black_pixel_cords.extend(get_middle_of_black_point_border(black_pixels_coords))
    black_pixels_coords = []

    return centered_black_pixel_cords

def bresenham_line(x0, y0, x1, y1):
    """Returns the list of points in the line from (x0, y0) to (x1, y1) using Bresenham's algorithm.
    param: x0 absolut kord
    param: y0 absolut kord
    param: x1 absolut kord
    param: y1 absolut kord
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points

def check_line_for_black_points(image, x0, y0, x1, y1):
    """
    Checks if all points on the line between (x0, y0) and (x1, y1) are black within a margin.

    :param image: NumPy array representing the image.
    :param x0: Global x-coordinate of the start point.
    :param y0: Global y-coordinate of the start point.
    :param x1: Global x-coordinate of the end point.
    :param y1: Global y-coordinate of the end point.
    :param margin: Margin for pixel values to be considered black.
    :return: True if all points on the line are black, False otherwise.
    """
    line_points = bresenham_line(x0, y0, x1, y1)
    margin = configure.black_pixel_margin
    black_threshold = np.array([margin, margin, margin])

    for x, y in line_points:
        # Check if the pixel is within image boundaries
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            pixel_value = image[y, x]
            #print(f"Checking pixel at ({x}, {y}) with value {pixel_value} against threshold {black_threshold}")
            if not np.all(pixel_value <= black_threshold):
                #print(f"Pixel at ({x}, {y}) is not black: {pixel_value}")
                return False
        else:
            #print(f"Pixel at ({x}, {y}) is out of image bounds")
            return False
    return True


def distance_point_to_line(x, y, x0, y0, x1, y1):
    """
    Calculate the perpendicular distance from point (x, y) to the line segment (x0, y0) - (x1, y1).
    """
    numerator = abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0)
    denominator = np.sqrt((y1 - y0)**2 + (x1 - x0)**2)
    distance = numerator / denominator
    return distance

def points_near_line(point_list, x0, y0, x1, y1, threshold_distance):
    """
    Check which points in point_list are near the line segment (x0, y0) - (x1, y1) within the threshold_distance.
    """
    near_points = []
    for (x, y) in point_list:
        distance = distance_point_to_line(x, y, x0, y0, x1, y1)
        if distance <= threshold_distance:
            near_points.append((x, y))
    return near_points

def get_middle_of_black_point_border(point_list:list):
    """
    Middles the black connect lines
    returns: List of (x_n,y_n) that are the middle of a black line
    """

    if not point_list:
        return []  # If the list is empty, return []

    # Extract all x-values and y-values
    x_values = [x for x, y in point_list]
    y_values = [y for x, y in point_list]

    # Check if all x-values are the same
    all_x_same = all(x == x_values[0] for x in x_values)
    all_y_same = all(y == y_values[0] for y in y_values)

    averaged_line_start= []
    if all_x_same:
        # Sort the y-values to find consecutive points
        sorted_y = sorted(y_values)
        for i in range(len(sorted_y) - 1):
            if sorted_y[i+1] - sorted_y[i] == 1:
                # If the points are consecutive, calculate the middle
                middle_y = (sorted_y[i] + sorted_y[i+1]) / 2
                averaged_line_start.append((x_values[0], middle_y))

    elif all_y_same:
        # Sort the x-values to find consecutive points
        sorted_x = sorted(x_values)
        for i in range(len(sorted_x) - 1):
            if sorted_x[i+1] - sorted_x[i] == 1:
                # If the points are consecutive, calculate the middle
                middle_x = (sorted_x[i] + sorted_x[i+1]) / 2
                averaged_line_start.append((middle_x, y_values[0]))

    else:
        # If neither all x nor all y values are the same, return an empty list
        return []
     
    averaged_line_start = [(int(x),int(y)) for (x,y) in averaged_line_start]
    return averaged_line_start


def test():
    dataset_name = ask_for_dataset(new_create_bool=False)
    train_set, val_set = load_datasets(dataset_name)
 
    # Load Image
    id = train_set.id_list[0]
    img = train_set.image_dic[id]
    data = train_set.label_dic[id]

    # Convert to right datatype
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    
    black_line_list = []
    debug =[]
    for i, ((x_0, y_0), class_id, degree) in enumerate(data):
        part_img, origin = cut_image_np_safe(img, x_0, y_0)
        border_black_points_list = check_for_black_pixel_at_border(part_img, origin)
        for (x_1, y_1) in border_black_points_list:
            if check_line_for_black_points(img, x_0, y_0, x_1, y_1):
                debug.extend([(x_0, y_0),(x_1, y_1)])
                black_line_list.append(((x_0, y_0),(x_1, y_1)))

    img = draw_stuff_on_image_and_save(img, debug,black_line_list)
    visualize_image(img)

if __name__ == '__main__':
    test()
