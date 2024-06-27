from PIL import Image
import ast,os, random, math
import numpy as np

def get_label_dic(path):
    labels = {}
    with open(f'{path}/label.txt','r') as label_file:
        data = label_file.read()
        data = data.split('\n')
        for element in data:
            split = element.split(':')
            if split[0] != '':
                key, value = split
                labels[key] = [ast.literal_eval(x) for x in value.split('|')]
    return labels

def write_to_label_file(labels,out_path):
    with open(f'{out_path}/label.txt','w') as label_file:
        for key in labels:
            label_file.write(f'{key}:')
            label_file.write('|'.join(str(item) for item in labels[key]))
            label_file.write('\n')

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

    #new_image = Image.new("RGB", (img.size[0]+int(img.size[0]/2),img.size[1]+int(img.size[1]/2)), (255,255,255))
    #mask_image = Image.new("L", (img.size[0]+int(img.size[0]/2),img.size[1]+int(img.size[1]/2)), 255)
    #new_image.paste(img, (0, 0), mask_image)

    for i, degree in enumerate(degrees):
        value[i][2] = degree - rand_degree   

    center = (img.size[0]/2 , img.size[1]/2)
    for i, (x, y) in enumerate(points):
        x,y = rotate_point(x, y, center[0], center[1], rand_degree)

        value[i][0] = (int(x), int(y))
    
    return img, value


def rotate_all_images(path,output_path):
    labels = get_label_dic(path)
    
    for key in labels:
        with Image.open(f'{path}/{key}.jpg') as img:
            img, value = rotate_image(img,labels[key])
            img.save(f'{output_path}/{key}.jpg')
        labels[key] = value
    write_to_label_file(labels,output_path)