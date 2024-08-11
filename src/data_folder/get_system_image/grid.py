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













class Lager():
    def __init__(self,type) -> None:
        self.type = type
        self.koordinats = (None,None)
        self.connections = []
    def __str__(self):
        return f"{self.type}"
    


def generate_grid(rows, cols):
    return np.array([[0 for _ in range(cols)] for _ in range(rows)])
def expand_grid(grid):
    pass


# Function to check if an index is at the edge
def is_edge(idx,shape):
    return idx[0] == 0 or idx[0] == shape[0] - 1 or idx[1] == 0 or idx[1] == shape[1] - 1
# Function to check if two indices are adjacent
def is_adjacent(idx1, idx2):
    return abs(idx1[0] - idx2[0]) + abs(idx1[1] - idx2[1]) == 1

def insert_array(base_array, insert_array, position):
    """
    Insert an array into another array at a specified position.
    If the insertion goes beyond the base array's bounds, expand with zeros.
    
    :param base_array: The original array (2D NumPy array)
    :param insert_array: The array to insert (2D NumPy array)
    :param position: Tuple (row, col) specifying where to insert the top-left corner of insert_array
    :return: The resulting array after insertion
    """
    base_height, base_width = base_array.shape
    insert_height, insert_width = insert_array.shape
    row, col = position
    #print(base_height, row, insert_height)
    # Calculate the new dimensions
    new_height = max(base_height, row + insert_height)
    new_width = max(base_width, col + insert_width)

    # Create a new array with zeros
    result = np.zeros((new_height, new_width), dtype=base_array.dtype)

    # Copy the base array into the new array
    result[:base_height, :base_width] = base_array

    # Insert the new array
    result[row:row+insert_height, col:col+insert_width] = insert_array

    return result
    
def format_grid_for_print(grid):
    def format_cell(cell):
        if cell is None:
            return '0'
        return str(cell)  # This will use your custom __str__ method

    return '\n'.join([' '.join(map(format_cell, row)) for row in grid])

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
        if is_adjacent(idx1, idx2) and (is_edge(idx1,shape) or is_edge(idx2,shape))
    ]

    anzahl_connection_point = random.randint(1,2) 
    if anzahl_connection_point == 1:
        edge_pairs = [edge_pairs[random.randint(0,1)]]
    return {'scheibe': scheiben_elemt, 'connection_points': edge_pairs}



def create_fachwerk(num_scheiben):
    fachwerk = []
    all_connection_points = []
    combined_array = np.array([[]])
    position = (0,0)

    for i in range(num_scheiben):
        scheibe = generate_scheibe()
        
        # Update the connection points to reflect the new position in the fachwerk
        updated_connection_points = [
            ((p1[0] + position[0], p1[1] + position[1]), (p2[0] + position[0], p2[1] + position[1]))
            for p1, p2 in scheibe['connection_points']
        ]
        
        # Add the updated scheibe to the fachwerk
        fachwerk.append({
            'scheibe': scheibe['scheibe'],
            'start_position': position,
            'connection_points': updated_connection_points
        })

        combined_array = insert_array(combined_array, scheibe['scheibe'],position)
        pos_positions = updated_connection_points[random.randint(0,len(updated_connection_points)-1)]
        position = pos_positions[random.randint(0,1)]

        print('-----------')
        print(combined_array)

    return {
        'combined_array': combined_array,
        'scheiben': fachwerk,
        'all_connection_points': all_connection_points
    }


def test_grid_generation():
    lager_data = {'type': 0,
                  'koordinats': None,
                  'connections': 
                    {   
                        'to': (0,0),
                        'type': 0
                        }
                  }
    cols = configure.generated_system_colums
    rows = configure.generated_system_rows

    # generates an array with o and dimension (cols,rows)
    grid = np.empty((rows, cols), dtype=object)
    start_row= int(rows//2)
    


    # Set Start Element to Lager
    lager_type = random.randint(2,4)
    grid[start_row,0] = Lager(lager_type)
    #create_section(grid,start_row,0)

    # Example usage
    fachwerk = create_fachwerk(5)  # Create a fachwerk with 5 scheiben
    print("Combined Array:")
    print(fachwerk['combined_array'])
    print("\nAll Connection Points:")
    print(fachwerk['all_connection_points'])
    print("Alle scheiben")
    print(fachwerk['scheiben'])



        


    









if __name__ == "__main__":

    end_grid, con_loc_list = generate_a_connected_grid(5, 5, PROB=0.4)
    print(con_loc_list)
    get_lengths(end_grid)
    print("End Grid:")
    for row in end_grid:
        print(row)

