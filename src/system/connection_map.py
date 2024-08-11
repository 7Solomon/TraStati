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
    id = data_set.id_list[4]
    img = data_set.image_dic[id]
    data = data_set.label_dic[id]

    # Convert to right datatype
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    points = [(x_0, y_0) for (x_0, y_0), class_id, degree in data]
    staebe = []
    debug = []
    for (x,y) in points: 
        for (x_1,y_1) in points:
            if x != x_1 and y != y_1:
                is_black, corrected_x0, corrected_y0, corrected_x1, corrected_y1 = check_and_correct_black_line(img, x, y, x_1, y_1)
                if is_black:
                    staebe.append(((corrected_x0,corrected_y0),(corrected_x1,corrected_y1)))
                else:
                    debug.append(((x,y),(x_1,y_1)))
                
   
    img = draw_stuff_on_image_and_save(img,[],staebe)
    img = draw_stuff_on_image_and_save(img,[],debug, line_color=(0,255,0))
    visualize_image(img)

    """for i, ((x_0, y_0), class_id, degree) in enumerate(data):
        part_img, origin = cut_image_np_safe(img, x_0, y_0) 
        points = check_for_black_pixel_at_border(part_img, origin)

        display_img = draw_stuff_on_image_and_save(img, points, [])
        visualize_image(display_img)"""



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



def is_line_black(image, x0, y0, x1, y1, black_threshold):
    line_points = bresenham_line(x0, y0, x1, y1)
    for x, y in line_points:
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            if not np.all(image[y, x] <= black_threshold):
                return False
        else:
            return False
    return True


def find_continuous_line(image, x_start, y_start, line_margin, black_threshold):
    if (0 + line_margin <= x_start < image.shape[1] - line_margin and
        0 + line_margin <= y_start < image.shape[0] - line_margin):
        
        roi = image[y_start-line_margin:y_start+line_margin, 
                    x_start-line_margin:x_start+line_margin]
        color_mask = np.all(roi <= black_threshold, axis=-1)
        #print(color_mask)

        
        if np.any(color_mask):
            # Find coordinates of black pixels
            black_pixel_positions = np.where(color_mask)
            #print(black_pixel_positions)
            y_coords, x_coords = np.where(color_mask)
            #print(y_coords)
            #print(y_coords)
            # Calculate average x and y
            avg_x = np.mean(x_coords)
            avg_y = np.mean(y_coords)
            
            #print(f'avg: {(avg_x,avg_y)}')
            #img = draw_stuff_on_image_and_save(image,[(x_start+avg_x,y_start+avg_y)],[],point_color=(0,150,0))
            #visualize_image(img)
            return (int(x_start+avg_x),int(y_start+avg_y))
    
    return None

def check_and_correct_black_line(image, x0, y0, x1, y1):
    margin = configure.black_pixel_margin
    line_margin = configure.line_margin
    black_threshold = np.array([margin, margin, margin])

    # First, check if the original line is black
    if is_line_black(image, x0, y0, x1, y1, black_threshold):
        return True, x0, y0, x1, y1

    # If not, look for a continuous black line nearby
    midpoint_x = int((x0 + x1) / 2)
    midpoint_y = int((y0 + y1) / 2)
    corrected_center_point= find_continuous_line(image, midpoint_x, midpoint_y, line_margin, black_threshold)


    if corrected_center_point:

        c_x, c_y = corrected_center_point
        delta_x = int((c_x - midpoint_x)//2)
        delta_y = int((c_y - midpoint_y )//2)
        
        new_x0, new_y0, new_x1, new_y1 = x0 + delta_x, y0 + delta_y, x1 + delta_x, y1 + delta_y
        # Recheck if the corrected line is black
        img = draw_stuff_on_image_and_save(image,[], [((new_x0, new_y0), (new_x1, new_y1))], line_color=(200,0,0))
        visualize_image(img)
        if is_line_black(image, new_x0, new_y0, new_x1, new_y1, black_threshold):
            print('corrected')
            return True, new_x0, new_y0, new_x1, new_y1

    # If no black line found or corrected line is not black
    return False, x0, y0, x1, y1


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
    line_margin = configure.line_margin
    black_threshold = np.array([margin, margin, margin])

    for x, y in line_points:
        # Check if the pixel is within image boundaries
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            pixel_value = image[y, x]
            #print(f"Checking pixel at ({x}, {y}) with value {pixel_value} against threshold {black_threshold}")
            if not np.all(pixel_value <= black_threshold):
                #print(f"Pixel at ({x}, {y}) is not black: {pixel_value}")
                return False
            # Check if not just line over or under
            else:
                x_start,y_start = line_points[int(len(line_points)//2)]    # Take the middle of the line

                if 0 + line_margin <= x_start < image.shape[1] - line_margin and 0 + line_margin <= y_start < image.shape[0] - line_margin:
                    region_of_interest = image[y_start-margin:y_start+margin, x_start-margin:x_start+margin]
    

                    color_mask = np.abs(region_of_interest) <= black_threshold 
                    kernel = np.ones((3,3), np.uint8)
                    closed = cv2.morphologyEx(color_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
                    
                    for i in range(4):  # Check all 4 sides
                        rotated = np.rot90(closed, i)
                        if np.any(np.all(rotated, axis=0)):
                            return True


                        
        else:
            #print(f"Pixel at ({x}, {y}) is out of image bounds")
            return False
    return True









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
