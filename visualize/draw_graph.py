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




def draw_stuff_on_image_and_save(img, points, degree_lines):
    # Create a figure and axis
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis('off')
    point_color = (1, 0, 0)
    line_color = (0, 0, 1) 
    
    for point in points:
        
        ax.scatter(point[0], point[1], color=point_color)

    for line in degree_lines:
        #print(line)
        #print([line[0][0], line[1][0]], [line[0][1], line[1][1]])
        ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], color=line_color)
    # Show the plot
    fig.canvas.draw()
    image_np = np.array(fig.canvas.renderer.buffer_rgba())

    # Convert the RGB image to BGR (OpenCV uses BGR)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)

    # Display the image using OpenCV
    cv2.imshow('Image with Drawn Stuff', image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the image
    plt.savefig('output.jpg')

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

def draw_some_item(idx,path = 'data_folder/test_dataloader/train/'):
    #print('_____ACHTUNG_____ du hast vergessen, dass negative zahlen dir deine visiualizierung verhauen')
    #print('!!!!')
    #print('!!!!')
    dict = init_hash_map(path)
    #print(dict['dba043ff5e2e'])
    #first_key, first_value = next(iter(dict.items()))
    some_key, some_value = list(dict.items())[idx]
    img, points, degree_lines = get_points_and_path(path,some_key,some_value)
    #print(degree_lines)
    print(points)
    draw_stuff_on_image_and_save(img, points,degree_lines)

def test_drawer(value,img):
    points,degrees = get_points_from_label(value)
    degree_lines = get_degree_lines(points, degrees)
    draw_stuff_on_image_and_save(img, points,degree_lines)