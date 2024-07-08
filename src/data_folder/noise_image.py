from PIL import Image
import numpy as np
import random,os,ast,math
import cv2

from src.visualize.draw_graph import draw_stuff_on_image_and_save


#cv2.GaussianBlur(image_eq, (5, 5), 0)
def point_is_red(point):
    if np.int32(point[0]) - np.int32(point[1])> 50 and np.int32(point[0]) - np.int32(point[2]) > 50:   # np.int32 Dass die Values nicht overflown
        return True
    else:
        return False


def get_random_white_tensor(shape):
    random_tensor = np.random.normal(170, 5, shape)

    random_tensor[:, :, 0] = random_tensor[:, :, 0] - 0
    random_tensor[:, :, 1] = random_tensor[:, :, 1] - 4
    random_tensor[:, :, 2] = random_tensor[:, :, 1] - 7

    random_tensor_int = random_tensor.astype(np.uint8)
    return random_tensor_int


def get_trapez(image, points, degree_line_points, distortion_degree=-15):
    original_points = np.float32([[0, 0], [image.shape[1] - 1, 0], [0, image.shape[0] - 1], [image.shape[1] - 1, image.shape[0] - 1]])
    distortion = int(distortion_degree * 0.01 * image.shape[1])
    new_points = np.float32([[0, 0], [image.shape[1] - 1, 0], [distortion, image.shape[0] - 1], [image.shape[1] - distortion - 1, image.shape[0] - 1]])

    matrix = cv2.getPerspectiveTransform(original_points, new_points)
    
    
    corrected_image = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))
    points_reshaped = np.float32(points).reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(points_reshaped, matrix)
    transformed_points = np.int32(transformed_points.reshape(-1, 2))
    
    degree_line_points_reshaped = np.float32(degree_line_points).reshape(-1, 1, 2)
    transformed_degree_line_points = cv2.perspectiveTransform(degree_line_points_reshaped, matrix)
    transformed_degree_line_points = transformed_degree_line_points.reshape(-1, 2)
    # Transformation der Label_points
    return corrected_image, transformed_points, transformed_degree_line_points


def loop_over_image(img_array):
    x,y,color_n = img_array.shape
    random_tensor = get_random_white_tensor(img_array.shape)
    for i in range(x):
        for j in range(y):
            if np.all(img_array[i][j] >= [210,210,210]) or point_is_red(img_array[i][j]):
                img_array[i][j] = random_tensor[i][j]
    return img_array




def randomize_image(img,value):

    points = [x[0] for x in value]
    degrees = [x[2] for x in value]

    degree_line_points = [(math.cos(math.radians(degree))*100,math.sin(math.radians(degree))*100) for degree in degrees]

    possible__blurrs = [(1,1),(3,3,),(5,5)]
    trapez_kor, gausian_blur = random.randint(-20,0), possible__blurrs[random.randint(0,len(possible__blurrs)-1)]

    image_array = np.array(img)
    image_array, points, degree_line_points = get_trapez(image_array,points, degree_line_points,trapez_kor)
    image_array = loop_over_image(image_array)
    image_array = cv2.GaussianBlur(image_array, gausian_blur, 100)
    

    img = Image.fromarray(image_array)
    points = [tuple(e) for e in points.tolist()]

    degrees = [int(np.degrees(np.arctan2(y/100, x/100))) for (x,y) in degree_line_points]
    value = [[points[i],*e[1:-1], degrees[i]] for i,e in enumerate(value)]
    return img, value

