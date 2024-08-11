import numpy as np
from scipy import stats
import cv2

import src.configure as configure
from src.neural_network_stuff.custome_dataset import CustomImageDataset
from src.visualize.visualize_image import visualize_image 
from src.visualize.draw_graph import draw_stuff_on_image_and_save

def efficient_distance(points):
    """
    distances[i,j] is giving the distance between the points
    """

    # Convert list of tuples to NumPy array
    points = np.array(points)
    
    # Calculate pairwise differences
    diff = points[:, np.newaxis] - points
    
    # Calculate Euclidean distances
    distances = np.sqrt(np.sum(diff**2, axis=2))    
    return distances

def normalized_direction_matrix(points):
    # Convert list of points to a numpy array
    points_array = np.array(points)
    
    # Create a 3D array of differences
    # This broadcasts the subtraction operation
    diff = points_array[:, np.newaxis, :] - points_array[np.newaxis, :, :]
    
    # Calculate magnitudes of the difference vectors
    magnitudes = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero by setting zero magnitudes to 1
    magnitudes = np.where(magnitudes != 0, magnitudes, 1)
    
    # Normalize the difference vectors
    normalized = diff / magnitudes[:, :, np.newaxis]
    
    # Set the diagonal to zero (direction vector to itself is undefined)
    np.fill_diagonal(normalized[:, :, 0], 0)
    np.fill_diagonal(normalized[:, :, 1], 0)

    
    return normalized


def look_for_standart_length(distances):
    print(distances)
    # Creates a mask for effiziency
    mask = np.ones_like(distances, dtype=bool)
    np.fill_diagonal(mask, False)  # Remove self-comparison


    multiple_saver = {}
    rows, cols = distances.shape
    for i in range(rows):
        for j in range(i + 1, cols):
            element = distances[i, j]
            if element != 0:  # Avoid division by zero
                
                fractions = [1/3,1/2,2/3,1]
                for fraction in fractions:
                    # To not look at these 
                    temp_mask = mask.copy()
                    temp_mask[i, j] = False
                    temp_mask[j, i] = False

                    # Bedingung für Multiple
                    multiples_mask = (
                        element*fraction % distances < configure.standart_length_margin
                        ) | (
                            distances - (element*fraction % distances) < configure.standart_length_margin
                        ) & temp_mask
                    
                    # Abfrage if Multiple
                    if np.any(multiples_mask):
                        multiple_elements = np.argwhere(multiples_mask)

                        information = [(_,distances[_[0],_[1]]//distances[i,j]*fraction) for _ in multiple_elements.tolist()]   # Anzahl an L 
                        multiple_saver[(i,j)] = ((information),distances[i,j]*fraction)


    standart_length_points = max(multiple_saver.items(), key=lambda item: len(item[1][0]))  # Get the element with the max number of Connections

    return  {'value': standart_length_points[1][1],
             'positions': standart_length_points[1][0],
             'refrence:':standart_length_points[0]
                }

def look_for_standart_direction(directions):
    result = np.full_like(directions, None, dtype=object)

    for axis in [0, 1]:  # 0 für x, 1 für y
        values = directions[:,:,axis]
        max_val, min_val = np.max(values), np.min(values)
        
        # Close 1, -1
        for val, sign in [(max_val, 1), (min_val, -1)]:
            mask = np.isclose(values, val, atol=configure.standart_direction_margin)
            result[:,:,axis] = np.where(mask, sign, result[:,:,axis])
        # Close Zero
        close_zero_mask = np.isclose(values, 0.0, atol=configure.standart_direction_margin)
        result[:, :, axis] = np.where(close_zero_mask, 0.0, result[:, :, axis])

    np.fill_diagonal(result[:, :, 0], None) 
    np.fill_diagonal(result[:, :, 1], None) 
    return result


def get_standart_length_of_system(data_set: CustomImageDataset):

    # Load Data
    id = data_set.id_list[1]
    data = data_set.label_dic[id]
    img = data_set.image_dic[id]

    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)    




    points = [(x_0,y_0) for (x_0, y_0), class_id, degree in data]
    
    distances = efficient_distance(points)
    directions = normalized_direction_matrix(points)
    
    stadart_directions = look_for_standart_direction(directions)
    mask = np.vectorize(lambda x: x is not None)(stadart_directions)
    indices = np.argwhere(mask)

    debug = [(points[_[0]],points[_[1]]) for _ in indices]
    img = draw_stuff_on_image_and_save(img,[],debug)
    visualize_image(img)


    #print(directions)
    
    #standart_length_object = look_for_standart_length(distances)
    #print(standart_length_object['value'])
    #print(standart_length_object['positions'])
    #debug =[(points[_[0][0]], points[ _[0][1]]) for _ in standart_length_object['positions']] 
    #img = draw_stuff_on_image_and_save(img,[],debug, point_color=(0,255,0))
    #img = draw_stuff_on_image_and_save(img,[],debug)
    #visualize_image(img)
    

