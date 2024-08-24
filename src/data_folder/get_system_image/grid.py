import random
import src.configure as configure

import numpy as np


connector_list = []


def explore_grid(grid, i, j, connected_points, PROB):
    rows, cols = len(grid), len(grid[0])

    if i < 0 or j < 0 or i >= rows or j >= cols or (i, j) in connected_points:
        return

    connected_points.append((i, j))

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Adjacent points

    for x, y in neighbors:
        ni, nj = i + x, j + y
        if 0 <= ni < rows and 0 <= nj < cols and random.random() < PROB:  
            return
        elif 0 <= ni < rows and 0 <= nj < cols:
            if [(ni, nj),(i,j)] not in connector_list:
                connector_list.append([(i,j),(ni,nj)])
             
            explore_grid(grid, ni, nj, connected_points, PROB)
    

def connect_all_points(grid, PROB):
    rows, cols = len(grid), len(grid[0])
    connected_points = []

    # Start from a random point
    start_row, start_col = int(rows/2), int(cols/2)
    explore_grid(grid, start_row, start_col, connected_points, PROB)

    return connected_points

def generate_a_connected_grid(rows, cols, PROB=0.4):
    # Reset global 
    global connector_list
    connector_list = []
    
    start_grid = generate_grid(rows, cols)
    end_grid = generate_grid(rows, cols)


    connected_line = connect_all_points(start_grid, PROB)

    for n, (i, j) in enumerate(connected_line):
        end_grid[i][j] = 1

    print(connector_list)
    return end_grid, connector_list






def get_lengths(grid):
    """
    grid: array of shape (3,3) or what was defined in the get_grid
    returns: an array with {index,koordinaten} Dict with len 9 
    """
    ABSTAND = configure.latex_abstand
    randomize = configure.randomize_images

    lager_liste = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if randomize and random.random() < 0.5:
                randomizer = random.randint(-1,1)
            else:
                randomizer = 0
            if i != 0 or j != 0:
                x = i * ABSTAND + randomizer * ABSTAND/2
                y = j * ABSTAND + randomizer * ABSTAND/2
            else:
                x = i * ABSTAND
                y = j * ABSTAND

            if grid[i][j] == 1:
                lager_liste.append({'index':(i,j),
                                    'koordinaten':(y,-x)})
    
    return lager_liste














def generate_grid(rows, cols):
    return np.array([[0 for _ in range(cols)] for _ in range(rows)])



# Function to check if an index is at the edge
def is_edge(idx:tuple,shape):
    return idx[0] == 0 or idx[0] == shape[0] - 1 or idx[1] == 0 or idx[1] == shape[1] - 1
# Function to check if two indices are adjacent
def is_adjacent(idx1:tuple, idx2:tuple):
    return abs(idx1[0] - idx2[0]) + abs(idx1[1] - idx2[1]) == 1
# Function that checks if diagonal
def is_diagonal(idx1:tuple, idx2:tuple):
    return abs(idx1[0] - idx2[0]) ==1 and abs(idx1[1] - idx2[1]) == 1
def get_opposite_edge(edge):
    return {'right': 'left', 'left': 'right', 'top': 'bot', 'bot': 'top'}[edge]

def get_edge(idx1,idx2,array):
    ### Doesnt take the two possible but rather takes just the first one !!! NOt good
    row1, col1 = idx1
    row2, col2 = idx2
    if col1 == array.shape[1] - 1 and col2 == array.shape[1] - 1  :  # If both are in the last Col
        return 'right'
    if col1 == 0 and col2 == 0:  # If it's the first col
        return 'left'
    if row1 == array.shape[0] - 1 and row2 == array.shape[0] - 1:  # If it's the last row
        return 'bot'
    if row1 == 0 and row2 == 0:  # If it's the first row
        return 'top'
    if 0 <= col1 < array.shape[0] and 0 <= row1 < array.shape[1] or 0 <= col2 < array.shape[0] and 0 <= row2 < array.shape[1]:
        return 'inside'
    raise ValueError("Not Implemented edge!")



def append_scheibe(scheibe):
    array = scheibe['data']
    index_pairs = scheibe['connection_points']
    staebe = scheibe['staebe']

    connect_staebe_between_scheiben = []
    connection_points = []
    all_staebe = staebe.copy()

    for idx1, idx2 in index_pairs:
        # Get new one to add
        new_scheibe = generate_scheibe()
        add_scheibe = new_scheibe['data']
        add_scheibe_connection = new_scheibe['connection_points']
        add_scheibe_staebe = new_scheibe['staebe']

        # Determine the edge of the base scheibe where we're connecting
        base_edge = get_edge(idx1, idx2, array)

        # Offset for updating coordinates
        offset_x, offset_y = 0, 0

        # Add the new scheibe based on the base_edge
        match base_edge:
            case 'right':
                new_col = np.zeros((array.shape[0], add_scheibe.shape[1]), dtype=array.dtype)
                new_col[idx1[0]:idx1[0] + add_scheibe.shape[0], :] = add_scheibe
                array = np.concatenate((array, new_col), axis=1)
                offset_x, offset_y = 0, array.shape[1] - add_scheibe.shape[1]
                
                # Add staebe between connection points
                for i in range(add_scheibe.shape[0]):
                    if array[idx1[0] + i, idx1[1]] != 0 and new_col[idx1[0] + i, 0] != 0:
                        connect_staebe_between_scheiben.append(((idx1[0] + i, idx1[1]), (idx1[0] + i, array.shape[1] - 1)))

            case 'left':
                new_col = np.zeros((array.shape[0], add_scheibe.shape[1]), dtype=array.dtype)
                new_col[idx1[0]:idx1[0] + add_scheibe.shape[0], :] = add_scheibe
                array = np.concatenate((new_col, array), axis=1)
                offset_x, offset_y = 0, add_scheibe.shape[1]
                
                # Add staebe between connection points
                for i in range(add_scheibe.shape[0]):
                    if new_col[idx1[0] + i, -1] != 0 and array[idx1[0] + i, add_scheibe.shape[1]] != 0:
                        connect_staebe_between_scheiben.append(((idx1[0] + i, add_scheibe.shape[1] - 1), (idx1[0] + i, add_scheibe.shape[1])))

            case 'top':
                new_row = np.zeros((add_scheibe.shape[0], array.shape[1]), dtype=array.dtype)
                new_row[:, idx1[1]:idx1[1] + add_scheibe.shape[1]] = add_scheibe
                array = np.concatenate((new_row, array), axis=0)
                offset_x, offset_y = add_scheibe.shape[0], 0
                
                # Add staebe between connection points
                for j in range(add_scheibe.shape[1]):
                    if new_row[-1, idx1[1] + j] != 0 and array[add_scheibe.shape[0], idx1[1] + j] != 0:
                        connect_staebe_between_scheiben.append(((add_scheibe.shape[0] - 1, idx1[1] + j), (add_scheibe.shape[0], idx1[1] + j)))

            case 'bot':
                new_row = np.zeros((add_scheibe.shape[0], array.shape[1]), dtype=array.dtype)
                new_row[:, idx1[1]:idx1[1] + add_scheibe.shape[1]] = add_scheibe
                array = np.concatenate((array, new_row), axis=0)
                offset_x, offset_y = array.shape[0] - add_scheibe.shape[0], 0
                
                # Add staebe between connection points
                for j in range(add_scheibe.shape[1]):
                    if array[-add_scheibe.shape[0] - 1, idx1[1] + j] != 0 and new_row[0, idx1[1] + j] != 0:
                        connect_staebe_between_scheiben.append(((array.shape[0] - add_scheibe.shape[0] - 1, idx1[1] + j), (array.shape[0] - add_scheibe.shape[0], idx1[1] + j)))

        # Update coordinates of add_scheibe_connection
        updated_add_scheibe_connection = [
            ((x1 + offset_x, y1 + offset_y), (x2 + offset_x, y2 + offset_y))
            for (x1, y1), (x2, y2) in add_scheibe_connection
        ]
        connection_points.extend(updated_add_scheibe_connection)

        # Update coordinates of add_scheibe_staebe
        updated_add_scheibe_staebe = [
            ((x1 + offset_x, y1 + offset_y), (x2 + offset_x, y2 + offset_y))
            for (x1, y1), (x2, y2) in add_scheibe_staebe
        ]
        all_staebe.extend(updated_add_scheibe_staebe)

    # Add the connecting staebe to all_staebe
    all_staebe.extend(connect_staebe_between_scheiben)

    # Update existing connection points
    updated_connection_points = [
        ((x1, y1), (x2, y2)) for (x1, y1), (x2, y2) in connection_points
        if 0 <= x1 < array.shape[0] and 0 <= y1 < array.shape[1] and
           0 <= x2 < array.shape[0] and 0 <= y2 < array.shape[1]
    ]

    return {
        'data': array, 
        'staebe': all_staebe, 
        'connection_points': updated_connection_points
    }

def rotate_scheibe_to_match_edge(edge, matrix, idxs1,idxs2):
    if get_edge(idxs1,idxs2,matrix) == get_opposite_edge(edge):
        return matrix
    for i in range(1,3):
        print(i)
        rotated = np.rot90(matrix, k=i)
        if get_edge(idxs1,idxs2,matrix) == get_opposite_edge(edge):
            return rotated


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



def create_fachwerk():
    s = generate_scheibe()
    s = append_scheibe(s)
    return s


def test_grid_generation():
    pass


        
