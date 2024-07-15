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


def look_for_standart_length(distances):

    # Creates a mask for effiziency
    mask = np.ones_like(distances, dtype=bool)
    np.fill_diagonal(mask, False)  # Remove self-comparison


    multiple_saver = {}
    rows, cols = distances.shape
    for i in range(rows):
        for j in range(i + 1, cols):
            element = distances[i, j]
            if element != 0:  # Avoid division by zero
                
                debug = [1/3,1/2,2/3,1]
                for fraction in debug:
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
                        #for element_index in multiple_elements:
                            #print(f"{element:.2f} at ({i},{j}) is a multiple of {distances[element_index[0], element_index[1]]:.2f} at ({element_index[0]},{element_index[1]})")
                        multiple_saver[(i,j)] = (multiple_elements.tolist(),distances[i,j]*fraction)

    print(multiple_saver)
    #standart_length_points = max(multiple_saver.items(), key=lambda item: len(item[1][0]))
    #return standart_length_points
    #print(distances[standart_length[0]])
    #for key, multis in multiple_saver.items():
    #    for (i,j) in multis:
    #        print(f'{key, distances[i,j]}')
def get_standart_length_of_system(data_set: CustomImageDataset):

    # Load Data
    id = data_set.id_list[0]
    data = data_set.label_dic[id]
    img = data_set.image_dic[id]

    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)    




    points = [(x_0,y_0) for (x_0, y_0), class_id, degree in data]
    distances = efficient_distance(points)
    
    look_for_standart_length(distances)
    #print([(points[debug[0][0]],debug[0][1])])
    #print([(points[_[0]],_[1]) for _ in debug[1]])
    
    ##img = draw_stuff_on_image_and_save(img,[],[(points[debug[0][0]],points[debug[0][1]])], point_color=(0,255,0))
    #img = draw_stuff_on_image_and_save(img,[], [(points[_[0]],points[_[1]]) for _ in debug[1]])
    #visualize_image(img)
    

