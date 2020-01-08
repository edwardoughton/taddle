import os
import numpy as np
from shapely.geometry import Point
import random

def get_polygon_download_locations(polygon, number=20, seed=7):
    """
        Samples 20 points from a polygon
    """
    random.seed(seed)

    points = []
    min_x, min_y, max_x, max_y = polygon.bounds
    i = 0
    while i < number:
        point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        if polygon.contains(point):
            points.append([point.y, point.x])
            i += 1
    return points  # returns list of lat/lon pairs
    

            