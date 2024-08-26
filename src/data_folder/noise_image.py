from PIL import Image
import numpy as np
import random,os,ast,math
import cv2

from src.visualize.draw_graph import draw_stuff_on_image_and_save
from src import configure


#cv2.GaussianBlur(image_eq, (5, 5), 0)
def point_is_red(point):
    if np.int32(point[0]) - np.int32(point[1])> 50 and np.int32(point[0]) - np.int32(point[2]) > 50:   # np.int32 Dass die Values nicht overflown
        return True
    else:
        return False


def generate_degree_line_points(degrees:list):
    """
    returns list of [((n1x1,n1y1),(n1x2,n1y2)),... n]
    """
    radius = configure.degree_lines_radius
    radians = np.radians(degrees)
    return list(zip(np.cos(radians) * radius, np.sin(radians) * radius))

def get_random_white_tensor(shape):
    random_tensor = np.random.normal(170, 5, shape)

    random_tensor[:, :, 0] = random_tensor[:, :, 0] - 0
    random_tensor[:, :, 1] = random_tensor[:, :, 1] - 4
    random_tensor[:, :, 2] = random_tensor[:, :, 1] - 7

    random_tensor_int = random_tensor.astype(np.uint8)
    return random_tensor_int


def get_trapez(image:np.ndarray, label:dict, distortion_degree:float):

    # Get distortion Matrix
    original_points = np.float32([[0, 0], [image.shape[1] - 1, 0], [0, image.shape[0] - 1], [image.shape[1] - 1, image.shape[0] - 1]])
    distortion = int(distortion_degree * 0.01 * image.shape[1])
    new_points = np.float32([[0, 0], [image.shape[1] - 1, 0], [distortion, image.shape[0] - 1], [image.shape[1] - distortion - 1, image.shape[0] - 1]])
    matrix = cv2.getPerspectiveTransform(original_points, new_points)
    
    # Correct image
    corrected_image = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))

    # Correct points
    points_reshaped = np.float32([e['koordinaten'] for e in label.values()]).reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(points_reshaped, matrix)
    transformed_points = np.int32(transformed_points.reshape(-1, 2))
    

    # Correct degree lines
    degree_line_points_reshaped = np.float32(generate_degree_line_points([e['rotation'] for e in label.values()])).reshape(-1, 1, 2)
    transformed_degree_line_points = cv2.perspectiveTransform(degree_line_points_reshaped, matrix)
    transformed_degree_line_points = transformed_degree_line_points.reshape(-1, 2)

    # Update label dictionary
    for (key, element), new_coord, new_degree_line in zip(label.items(), transformed_points, transformed_degree_line_points):
        element['koordinaten'] = tuple(new_coord)
        element['transformed_degree_line'] = tuple(new_degree_line)
    
    # Transformation der Label_points
    return corrected_image, label


def loop_over_image(img_array):
    x,y,color_n = img_array.shape
    random_tensor = get_random_white_tensor(img_array.shape)
    for i in range(x):
        for j in range(y):
            if np.all(img_array[i][j] >= [210,210,210]) or point_is_red(img_array[i][j]):
                img_array[i][j] = random_tensor[i][j]
    return img_array




def randomize_image(img,label:dict):


    # Get Blurr and trapez Variables
    possible_blurrs = configure.possible_blurrs
    trapez_kor = random.randint(configure.trapez_kor_grenze_start,configure.trapez_kor_grenze_end)
    
    # Define blurr
    gausian_blur = possible_blurrs[random.randint(0,len(possible_blurrs)-1)]

    # Convert to Trapez
    image_array = np.array(img)
    image_array, label = get_trapez(image_array,label,trapez_kor)
    
    # Convert to noise
    image_array = loop_over_image(image_array)
    image_array = cv2.GaussianBlur(image_array, gausian_blur, 100)
    
    return Image.fromarray(image_array), label

