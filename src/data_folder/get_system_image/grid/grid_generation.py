import random
import src.configure as configure

from src.data_folder.get_system_image.grid.functions_for_grid_generation import *

import numpy as np


def count_nodes_connections_and_reactions(array, staebe, reactions):
    """
    Count the number of nodes, connections, and reactions.
    array: 2D numpy array representing the grid
    staebe: list of tuples representing connections
    reactions: number of external reactions or constraints
    returns: tuple (number_of_nodes, number_of_connections, number_of_reactions)
    """
    number_of_nodes = np.count_nonzero(array == 1)
    number_of_connections = len(staebe)
    number_of_reactions = reactions  # This should be provided based on your system constraints
    return number_of_nodes, number_of_connections, number_of_reactions

def is_statical_determinate_for_fachwerk(system):
    array = system['data']
    staebe = system['staebe']
    reactions = 3
    number_of_nodes, number_of_connections, number_of_reactions = count_nodes_connections_and_reactions(array, staebe, reactions)
    # Apply Gruebler's formula
    F = 3 * number_of_nodes - 2 * number_of_connections - number_of_reactions
    return F == 0

def generate_scheibe():
    """
    Creates an random array with one zero [[0,1],[1,1]]
    returns: this array in [scheibe] und die möglichen anknüpfunkte in [connection_points]
    """
    # init
    scheiben_elemt = np.array([[1,1],[0,1]])

    # Randomize
    flattened = scheiben_elemt.flatten()
    np.random.shuffle(flattened)
    scheiben_elemt = flattened.reshape(scheiben_elemt.shape)

    # For Edge detection
    shape = scheiben_elemt.shape 
    one_indices = np.where(scheiben_elemt != 0)
    indices = list(zip(*one_indices))

    
    # Get adjacent pairs at the edge
    edge_pairs = [
        (idx1, idx2) for i, idx1 in enumerate(indices)
        for j, idx2 in enumerate(indices[i+1:], i+1)
        if (is_adjacent(idx1, idx2) or is_diagonal(idx1, idx2)) and (is_edge(idx1,shape) or is_edge(idx2,shape))
    ]
    
    
    # Get the staebe
    staeb_pairs = [
        (idx1, idx2) for i, idx1 in enumerate(indices)
        for j, idx2 in enumerate(indices[i+1:], i+1)
        if is_adjacent(idx1,idx2) or is_diagonal(idx1,idx2)
    ]
    
    return {
            'data': scheiben_elemt,
             'staebe': staeb_pairs, 
             'connection_points': edge_pairs
            }

import numpy as np

def test_append(scheibe):
    array = scheibe['data']
    staebe = scheibe['staebe']
    connection_points = scheibe['connection_points']

    # Expand the array with padding for easier boundary handling
    new_array = np.pad(array, pad_width=1, mode='constant', constant_values=0)
    
    # Update the coordinates for staebe and connection_points after padding
    staebe = [((x1 + 1, y1 + 1), (x2 + 1, y2 + 1)) for (x1, y1), (x2, y2) in staebe]
    connection_points = [((x1 + 1, y1 + 1), (x2 + 1, y2 + 1)) for (x1, y1), (x2, y2) in connection_points]
    
    # Collect potential new staebe to add based on the connection points
    possible_new_staebe = []
    for _1, _2 in connection_points:
        possible_new_staebe.extend(get_zeros_next_to_connection((_1, _2), new_array))
    
    # Remove duplicates and ensure order is preserved
    possible_new_staebe = list(dict.fromkeys(possible_new_staebe))
    
    # Filter out invalid staebe
    filtered_new_staebe = [
        (node_idx, next_node)
        for node_idx, next_node in possible_new_staebe
        if not is_diagonal_crossing(node_idx, next_node, staebe) and is_adjacent(node_idx, next_node)
    ]
    
    # Add valid new staebe to the existing staebe list
    staebe.extend(filtered_new_staebe)
    
    # Ensure every node has exactly 2 connections
    staebe = ensure_node_connections(staebe)
    
    # Update array with new staebe nodes
    for node_idx in {s[0] for s in filtered_new_staebe}:
        new_array[node_idx] = 1
    
    # Ensure all nodes are connected (simple connectivity check)
    ensure_connectivity(new_array, staebe)
    
    return {
        'data': new_array,
        'staebe': staebe,
        'connection_points': None
    }


def create_fachwerk():
    ### Append scheibe mach key ERROR da 0 als knotten genommen wird???
    s = generate_scheibe()
    #s = test_append(s)
    #print(s)
    
    #s = append_scheibe(s)
    return s


def test_grid_generation():
    pass


        
