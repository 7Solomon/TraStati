import numpy as np
from src.data_folder.get_system_image.convert_sytem_as_png import convert_system
from src.data_folder.get_system_image.save_system_to_tex_file import getSystemAndSave, loopSystem
from src.data_folder.manage_datasets import create_random_image

from src.data_folder.get_system_image.grid import create_fachwerk


from src.visualize.visualize_image import visualize_image
from src.visualize.draw_graph import draw_stuff_on_image_and_save, get_degree_lines

from src.data_folder.resize_image import find_zero_size




def distance_point_to_line(x, y, x0, y0, x1, y1):
    """
    Calculate the perpendicular distance from point (x, y) to the line segment (x0, y0) - (x1, y1).
    """
    numerator = abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0)
    denominator = np.sqrt((y1 - y0)**2 + (x1 - x0)**2)
    distance = numerator / denominator
    return distance

def points_near_line(point_list, x0, y0, x1, y1, threshold_distance):
    """
    Check which points in point_list are near the line segment (x0, y0) - (x1, y1) within the threshold_distance.
    """
    near_points = []
    for (x, y) in point_list:
        distance = distance_point_to_line(x, y, x0, y0, x1, y1)
        if distance <= threshold_distance:
            near_points.append((x, y))
    return near_points





def test():

    img, id, label = create_random_image()
    visualize_image(img)
    

if __name__ == '__main__':
    test()
