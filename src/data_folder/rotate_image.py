from PIL import Image
import ast,os, random, math
import numpy as np

def rotate_point(x, y, cx, cy, theta):
    # Convert angle to radians
    theta_rad = math.radians(theta)

    # Calculate the new coordinates with adjusted signs
    x_new = (x - cx) * math.cos(theta_rad) + (y - cy) * math.sin(theta_rad) + cx
    y_new = -(x - cx) * math.sin(theta_rad) + (y - cy) * math.cos(theta_rad) + cy

    return (x_new, y_new)

def rotate_image(img:Image.Image, value:dict):
    #points = [e[] for e in value]
    #degrees = [e[2] for e in value]

    rand_degree = random.randint(0,360)  # Könnte zu probelmen bei Loss führen
    img = img.rotate(rand_degree,fillcolor = (255,255,255))
    center = (img.size[0]/2 , img.size[1]/2)

    new_label = {}
    for key, element in value.items():
        # Rotate Degree
        new_rotation = element['rotation'] - rand_degree   

        #Rotate koordinats
        (x,y) = element['koordinaten']
        (x,y) = rotate_point(x, y, center[0], center[1], rand_degree)

        new_label[key] = element.copy()
        new_label[key]['koordinaten'] = (int(x),int(y))
        new_label[key]['rotation'] = new_rotation
    
    return img, new_label

