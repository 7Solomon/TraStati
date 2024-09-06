import numpy as np

from collections import defaultdict

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
def get_neighbor_idx(idx:tuple, shape:tuple)->list:
    row, col = idx
    num_rows, num_cols = shape
    neighbors = []
    
    #### ADD Edge/oriantatioin detection  and then adjust deltas for them
    deltas = [(-1, -1), (-1, 0), (-1, 1),
                  ( 0, -1),          ( 0, 1),
                  ( 1, -1), ( 1, 0), ( 1, 1)]
    for dr, dc in deltas:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < num_rows and 0 <= new_col < num_cols:
            neighbors.append((new_row, new_col))
    return neighbors

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
def get_zeros_next_to_connection(connection, array):
    (x1, y1), (x2, y2) = connection
    # List of positions to check around the connection
    adjacent_positions = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if (dx != 0 or dy != 0):
                new_x1, new_y1 = x1 + dx, y1 + dy
                new_x2, new_y2 = x2 + dx, y2 + dy
                if array[new_x1, new_y1] == 0:
                    adjacent_positions.append(((new_x1, new_y1), (x2, y2)))
                if array[new_x2, new_y2] == 0:
                    adjacent_positions.append(((x1, y1), (new_x2, new_y2)))
    return adjacent_positions

# Ensure no diagonal crossings
def is_diagonal_crossing(p1, p2, staebe):
    for (x1, y1), (x2, y2) in staebe:
        if (x1 != x2 and y1 != y2) and (
            min(p1[0], p2[0]) < max(x1, x2) and max(p1[0], p2[0]) > min(x1, x2) and
            min(p1[1], p2[1]) < max(y1, y2) and max(p1[1], p2[1]) > min(y1, y2)
        ):
            return True
    return False

def ensure_connectivity(array, staebe):
    visited = set()
    
    def dfs(node):
        stack = [node]
        while stack:
            n = stack.pop()
            if n not in visited:
                visited.add(n)
                for (n1, n2) in staebe:
                    if n == n1 and n2 not in visited:
                        stack.append(n2)
                    elif n == n2 and n1 not in visited:
                        stack.append(n1)
    
    start_node = (1, 1)
    dfs(start_node)
    
    all_nodes = set(tuple(node) for node in np.argwhere(array == 1))
    missing_nodes = all_nodes - visited
    
    if missing_nodes:
        for node in missing_nodes:
            for existing_node in all_nodes:
                if existing_node != node and is_adjacent(node, existing_node):
                    staebe.append((node, existing_node))
                    break

def ensure_node_connections(staebe):
    # Create a dictionary to count connections for each node
    connection_count = defaultdict(int)
    
    for (n1, n2) in staebe:
        connection_count[n1] += 1
        connection_count[n2] += 1
    
    new_staebe = []
    nodes_with_one_connection = [node for node, count in connection_count.items() if count == 1]
    
    for node in nodes_with_one_connection:
        # Find a node with 0 connections to connect to
        for other_node in connection_count:
            if connection_count[other_node] < 2 and node != other_node:
                new_staebe.append((node, other_node))
                connection_count[node] += 1
                connection_count[other_node] += 1
                break
    
    # Add new connections to the existing staebe
    staebe.extend(new_staebe)
    
    return staebe