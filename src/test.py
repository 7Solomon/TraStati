


def bresenham_line(x0, y0, x1, y1):
    """Returns the list of points in the line from (x0, y0) to (x1, y1) using Bresenham's algorithm.
    param: x0 absolut kord
    param: y0 absolut kord
    param: x1 absolut kord
    param: y1 absolut kord
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points

def check_line_for_black_points(image, x0, y0, x1, y1):
    """
    Checks if all points on the line between (x0, y0) and (x1, y1) are black within a margin.

    :param image: NumPy array representing the image.
    :param x0: Global x-coordinate of the start point.
    :param y0: Global y-coordinate of the start point.
    :param x1: Global x-coordinate of the end point.
    :param y1: Global y-coordinate of the end point.
    :param margin: Margin for pixel values to be considered black.
    :return: True if all points on the line are black, False otherwise.
    """
    line_points = bresenham_line(x0, y0, x1, y1)
    margin = configure.black_pixel_margin
    black_threshold = np.array([margin, margin, margin])

    for x, y in line_points:
        # Check if the pixel is within image boundaries
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            pixel_value = image[y, x]
            #print(f"Checking pixel at ({x}, {y}) with value {pixel_value} against threshold {black_threshold}")
            if not np.all(pixel_value <= black_threshold):
                #print(f"Pixel at ({x}, {y}) is not black: {pixel_value}")
                return False
        else:
            #print(f"Pixel at ({x}, {y}) is out of image bounds")
            return False
    return True


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
    pass

        
        #debug_2.extend(border_black_points_list)
        #for (x_1, y_1) in border_black_points_list:
        #    if check_line_for_black_points(img, x_0, y_0, x_1, y_1):
        #        debug.extend([(x_0, y_0),(x_1, y_1)])
        #        black_line_list.append(((x_0, y_0),(x_1, y_1)))

    

if __name__ == '__main__':
    test()
