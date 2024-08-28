import re, os
import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError
import matplotlib.pyplot as plt


from src import configure


def find_zero_size(image):
    """
    Finds red dot in Image and returns first position, of its encounter
    """
    # Convert image to numpy array
    img_array = np.array(image)
    
    # Create a boolean mask for the condition
    mask = (img_array[:, :, 2] < 100) & (img_array[:, :, 0] > 200)
    
    # Find the indices where the condition is true
    indices = np.argwhere(mask)
    print(tuple(indices[0]))
    print(indices.size)

    # Return the first matching position, if any
    return tuple(indices[0]) if indices.size > 0 else None

            

            
def found_last_black_pixel(img):
    for x in range(img.size[0]):
            for y in range(img.size[1]):
                        
                pixel_position = (x,y)

                color_value = img.getpixel(pixel_position)

                if color_value != 255:
                    print(f"Found pixel at {pixel_position}, with color values {color_value}")


def paint_points(img,label):
    zero_zero = (1045, 1438)
    draw = ImageDraw.Draw(img)

    for element in label:
        (x,y) = element[0]
        point = (zero_zero[0]+int(x*12),zero_zero[1]+int(abs(y)*12))
        draw.point(point, fill=(0, 255, 0))
    
    draw._image.show()

    
def cut(img: Image.Image, pos: tuple, label: dict) -> Image.Image:
    """
    Cut image to the size of the system plus ABSTAND.
    
    Args:
    img (Image.Image): The input image.
    pos (tuple): The starting position (x, y).
    label (dict): Dictionary containing element positions.
    
    Returns:
    Image.Image: The cropped image.
    """
    ABSTAND = configure.cut_image_margin

    # Convert label positions to numpy array for efficient operations
    points = np.array([(pos[0] + int(x*12), pos[1] + int(abs(y)*12)) 
                   for element in label.values()
                   for x, y in [element['koordinaten']]])

    # Find max points efficiently using numpy
    max_point_x, max_point_y = np.max(points, axis=0)

    # Calculate crop boundaries
    left = max(0, int(pos[0]) - ABSTAND)
    top = max(0, int(pos[1]) - ABSTAND)
    right = min(img.width, int(max_point_x) + ABSTAND)
    bottom = min(img.height, int(max_point_y) + ABSTAND)

    # Crop image
    return img.crop((left, top, right, bottom))


    

def create_label_for_cut_images(label_data: dict):
    ABSTAND = configure.cut_image_margin
    
    # Loop over items and change position data
    new_label_data = {}
    for key, element in label_data.items():
        x, y = element['koordinaten']
        new_point = (ABSTAND + int(x*12), ABSTAND + int(abs(y)*12))
        
        new_element = element.copy()
        new_element['koordinaten'] = new_point
        
        new_label_data[key] = new_element

    return new_label_data

def resize(img, label):
    try:
        position = find_zero_size(img)
        
        # Switch position for PIL and numpy missmatch 
        position = (position[1],position[0])

        cut_image = cut(img, position, label)


        label = create_label_for_cut_images(label)
        return cut_image, label
             
    except UnidentifiedImageError:
        return None


"""
def resize_all_images():
    paths = os.listdir('src/data_folder/get_system_image/img')
    for path in paths:
        if path != 'label.txt':
            id = path.split('.')[0]
            resize(id)
"""



if __name__ == "__main__": 
    print('no right filo')
    #resize()