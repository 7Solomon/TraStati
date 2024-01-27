import random


connector_list = []

def generate_grid(rows, cols):
    return [[0 for _ in range(cols)] for _ in range(rows)]

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

def get_lengths(grid, ABSTAND = 20, randomize = True):
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
                lager_liste.append([(i,j),(y,-x)])
                #print(f'x,y: {(x,y)}')
    return lager_liste
    



if __name__ == "__main__":

    end_grid, con_loc_list = generate_a_connected_grid(5, 5, PROB=0.4)
    print(con_loc_list)
    get_lengths(end_grid)
    print("End Grid:")
    for row in end_grid:
        print(row)

