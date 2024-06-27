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

    # Top border
    for x in range(img_part.shape[1]):
        if np.all(img_part[0, x] <= black_threshold):
            black_pixels_coords.append((origin_x + x, origin_y))

    # Bottom border
    for x in range(img_part.shape[1]):
        if np.all(img_part[-1, x] <= black_threshold):
            black_pixels_coords.append((origin_x + x, origin_y + img_part.shape[0] - 1))

    # Left border
    for y in range(img_part.shape[0]):
        if np.all(img_part[y, 0] <= black_threshold):
            black_pixels_coords.append((origin_x, origin_y + y))

    # Right border
    for y in range(img_part.shape[0]):
        if np.all(img_part[y, -1] <= black_threshold):
            black_pixels_coords.append((origin_x + img_part.shape[1] - 1, origin_y + y))

    return black_pixels_coords

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
            print(f"Checking pixel at ({x}, {y}) with value {pixel_value} against threshold {black_threshold}")
            if not np.all(pixel_value <= black_threshold):
                print(f"Pixel at ({x}, {y}) is not black: {pixel_value}")
                return False
        else:
            print(f"Pixel at ({x}, {y}) is out of image bounds")
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

# Example usage:
line_start = (100, 100)
line_end = (300, 300)
threshold_distance = 10
point_list = [(120, 110), (200, 200), (310, 310), (280, 290)]

near_points = points_near_line(point_list, line_start[0], line_start[1], line_end[0], line_end[1], threshold_distance)

print("Points near the line:", near_points)

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
    for (x, y), class_id, degree in data:
        part_img, origin = cut_image_np_safe(img, x, y)
        border_black_points_list = check_for_black_pixel_at_border(part_img, origin)

        # Calculate the line 
        for (x_1, y_1) in border_black_points_list:
            if check_line_for_black_points(img, x, y, x_1, y_1):
                print(f"Line from ({x}, {y}) to ({x_1}, {y_1}) is valid.")
                black_line_list.append([(x, y), (x_1, y_1)])
            else:
                print(f"Line from ({x}, {y}) to ({x_1}, {y_1}) is invalid.")
            
    img = draw_stuff_on_image_and_save(img, [], black_line_list)
    visualize_image(img)

if __name__ == '__main__':
    test()
