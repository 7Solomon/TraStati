import re, os

from PIL import Image, ImageDraw, UnidentifiedImageError
import matplotlib.pyplot as plt

from src.visualize.draw_graph import test_drawer

def find_zero_size(image):
    #if image.getpixel((1088, 1361)) == (252,0,0):
    #    print(f"Found pixel at (1088, 1361) as expected)")
    #    return (1088, 1361)
    #else:
    for x in range(image.size[0]):
        for y in range(image.size[1]):
                    
            pixel_position = (x,y)
            color_value = image.getpixel(pixel_position)
            if color_value[2] < 100 and color_value[0]>200:
                    
                return pixel_position
            #print(f"Found pixel at {pixel_position}, with color values {color_value}")

            
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

    


def cut(img, pos, label, ABSTAND = 200):
    max_point_x,max_point_y = 0,0
    for element in label:
        (x,y) = element[0]
        point = (pos[0]+int(x*12),pos[1]+int(abs(y)*12))
        #print(point)
        if point[0] > max_point_x:
            max_point_x = point[0]
        if point[1] > max_point_y:
            max_point_y = point[1]
    #print((max_point_x,max_point_y))

    left = int(pos[0]) - ABSTAND
    top = int(pos[1]) - ABSTAND
    right = int(max_point_x) + ABSTAND
    bot = int(max_point_y)+ ABSTAND

    cropped_img = img.crop((left, top, right, bot))
    return cropped_img
    #print((left,top,right,bot))
    # Crop and display the image200
# Example usage:
# cut(img, (1088, 1361), [(0, 0), (20.0, -0.0), (0.0, -20.0), (20.0, -20.0), (20.0, -40.0)])

    


def create_label_for_cut_images(label_data, ABSTAND = 200):
    for i,element in enumerate(label_data):
        (x,y) = element[0]
        point = (ABSTAND + int(x*12),ABSTAND + int(abs(y)*12))
        label_data[i] = [(point),*element[1:]]

    return label_data

def resize(img, label):
    try:
        position = find_zero_size(img)
        cut_image = cut(img, position, label, ABSTAND=300)
        label = create_label_for_cut_images(label, ABSTAND = 300)
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