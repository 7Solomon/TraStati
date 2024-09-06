import numpy as np

from src import configure
def get_lengths(grid):
    """
    grid: array of shape (n,n) or what was defined in the get_grid
    returns: an array with {index, koordinaten} Dict with len 9 
    """
    ABSTAND = configure.latex_abstand  # Assuming this is defined somewhere in your config
    randomize = configure.randomize_images  # Assuming this is defined somewhere in your config

    grid = np.array(grid)  # Convert the grid to a NumPy array if it isn't already

    # Generate index grids
    indices = np.indices(grid.shape)
    i = indices[0]
    j = indices[1]

    # Apply randomization if necessary
    if randomize:
        randomizer = np.random.randint(-1, 2, size=grid.shape)  # Random values in [-1, 0, 1]
        randomizer = np.where(np.random.random(grid.shape) < 0.5, randomizer, 0)  # Apply randomizer conditionally
    else:
        randomizer = np.zeros(grid.shape, dtype=int)

    # Calculate coordinates
    x = i * ABSTAND + randomizer * ABSTAND / 2
    y = j * ABSTAND + randomizer * ABSTAND / 2

    # Set the first coordinate (0,0) to avoid randomization
    x[0, 0] = 0
    y[0, 0] = 0

    # Create a list of dictionaries for where grid[i][j] == 1
    lager_liste = [
        {'index': (i_, j_), 'koordinaten': (y_, -x_)}
        for i_, j_, x_, y_ in zip(i.flat, j.flat, x.flat, y.flat) if grid[i_, j_] == 1
    ]

    return lager_liste


