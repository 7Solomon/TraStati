from PIL import Image
import ast,os, random, math
import numpy as np

def rotate_point(x, y, cx, cy, theta):
    # Convert angle to radians
    theta_rad = math.radians(theta)

    # Calculate the new coordinates with adjusted signs
    x_new = (x - cx) * math.cos(theta_rad) + (y - cy) * math.sin(theta_rad) + cx
    y_new = -(x - cx) * math.sin(theta_rad) + (y - cy) * math.cos(theta_rad) + cy

    return x_new, y_new

def rotate_image(img, value):
    points = [e[0] for e in value]
    degrees = [e[2] for e in value]
    rand_degree = random.randint(0,360)  # Könnte zu probelmen bei Loss führen
    
    img = img.rotate(rand_degree,fillcolor = (255,255,255))

    for i, degree in enumerate(degrees):
        value[i][2] = degree - rand_degree   

    center = (img.size[0]/2 , img.size[1]/2)
    for i, (x, y) in enumerate(points):
        x,y = rotate_point(x, y, center[0], center[1], rand_degree)

        value[i][0] = (int(x), int(y))
    
    return img, value

