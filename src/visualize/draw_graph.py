import ast, math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import cv2


def init_hash_map(path):
    hash_map = {}
    with open(f'{path}label.txt','r') as file:
        data = file.read().split('\n')
        for line in data:
            id= line.split(':')[0]  
            if id != '':
                value = line.split(':')[1]
                hash_map[id] = [ast.literal_eval(e) for e in value.split('|')]
    return hash_map




def draw_stuff_on_image_and_save(img, points: list, degree_lines:list , point_color:tuple = (255, 0, 0), line_color:tuple = (0, 0, 255), save:bool = False, name='default'):
    """
    Zeichnet punkte und linien auf das Image
    return Image mit aufgemalten punkten und linien 
    """
    # Converts pil image in array
    if type(img) is Image.Image:
        img = np.array(img)


    # Points zu ints 
    try:
        points = [(int(x),int(y)) for (x,y) in points]
    except:
        print('Draw Stuff on image did not work, points can not be INTED!')
        return img
    
    # Loop over points to draw
    for point in points:
        cv2.circle(img, point, radius=5, color=point_color, thickness=-1)

    

    # Degree Lines zu Ints
    try:
        degree_lines = [((int(x_0), int(y_0)),(int(x_1), int(y_1))) for ((x_0, y_0),(x_1, y_1)) in degree_lines]
    except:
        print('Draw Stuff on image did not work, Degree Lines can not be INTED!')
        return img 

    # Loop over lines to draw
    for line in degree_lines:
        cv2.line(img, line[0], line[1], line_color, 2)



    # Convert the RGB image to BGR (OpenCV uses BGR)
    ##img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    ## Save the image
    if save:
        plt.savefig(f'drawn_onto_image_{name}.jpg')
    
    return img

    

def get_degree_lines(points, degrees):
    degree_lines = []
    line_length = 100

    assert len(points) == len(degrees)

    for i,degree in enumerate(degrees):
        # Check for Gelenke
        if degree is None:
            continue

        # transform degree
        degree = degree + 270
        #print(math.radians(degree),math.cos(math.radians(degree)))
        #print(f'test:{math.sin(math.radians(84)),math.sin(math.radians(-84))}')
        ax = math.cos(math.radians(degree))*1
        ay = math.sin(math.radians(degree))*1
        #print(ax,ay)
        #print(math.cos(math.radians(degree))*1,math.sin(math.radians(degree))*1)
        line_point_1 = (int(points[i][0] + ax*line_length), int(points[i][1]+ay*line_length))
        line_point_2 = (int(points[i][0] - ax*line_length), int(points[i][1]-ay*line_length))
        degree_lines.append((line_point_1, line_point_2))
    return degree_lines

def get_points_from_label(value):
    """
    Outdated not needed anymore
    """
    if isinstance(value[0], tuple):
        #print(value[2])
        points = [value[0]]
        degrees = [value[2] if value[1] != 0 else None]

    else:
        #print([f'type: {e[1]}|{e[2]}' for e in value])
        points = [e[0] for e in value]
        degrees = [e[2] if e[1] != 0 else None for e in value]
    return points,degrees

def get_points_and_path(path,id,value):
    img = Image.open(f'{path}{id}.jpg')

    points,degrees = get_points_from_label(value)
    degree_lines = get_degree_lines(points, degrees)

    return img, points, degree_lines


