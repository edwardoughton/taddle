"""
Create 1km grid.

Written by Ed Oughton

Winter 2020

"""
import os
import configparser

import geopandas as gpd
from shapely.geometry import Polygon
import pandas as pd
import numpy as np

from download_images import get_polygon_download_locations #, ImageDownloader

# CONFIG = configparser.ConfigParser()
# CONFIG.read(os.path.join(os.path.dirname(__file__), 'script_config.ini'))
# BASE_PATH = CONFIG['file_locations']['base_path']

# DATA_RAW = os.path.join(BASE_PATH, 'raw')
# DATA_PROCESSED = os.path.join(BASE_PATH, 'processed')

DATA_PROCESSED = 'data/malawi/grid'
DATA_RAW = 'data/malawi/shapefile'

def generate_grid(country):

    filename = 'national_outline_{}.shp'.format(country)
    country_outline = gpd.read_file(os.path.join(DATA_RAW, filename))

    country_outline.crs = {'init':'epsg:4326'}
    country_outline = country_outline.to_crs({'init':'epsg:3857'})

    xmin,ymin,xmax,ymax = country_outline.total_bounds

    length = 1e4 #1000
    wide = 1e4 #1000

    cols = list(range(int(np.floor(xmin)), int(np.ceil(xmax)), int(wide)))
    rows = list(range(int(np.floor(ymin)), int(np.ceil(ymax)), int(length)))
    rows.reverse()

    polygons = []
    for x in cols:
        for y in rows:
            polygons.append( Polygon([(x,y), (x+wide, y), (x+wide, y-length), (x, y-length)]))

    grid = gpd.GeoDataFrame({'geometry':polygons})
    intersection = gpd.overlay(grid, country_outline, how='intersection')
    intersection.crs = {'init' :'epsg:3857'}
    intersection = intersection.to_crs({'init': 'epsg:4326'})

    intersection.to_file(os.path.join(DATA_PROCESSED, 'grid.shp'))

    return print('Completed grid generation process')


def generate_country_download_locations(country, num_per_grid=20):
    grid = gpd.read_file(os.path.join(DATA_PROCESSED, 'grid.shp'))
    lat_lon_pairs = grid['geometry'].apply(lambda polygon: get_polygon_download_locations(polygon, number=num_per_grid))
    centroids = grid['geometry'].centroid

    with open(os.path.join(DATA_PROCESSED, 'image_download_locs.txt'), 'w') as f:
        columns = ['centroid_lat', 'centroid_lon', 'image_lat', 'image_lon']
        f.write(','.join(columns) + '\n')

        for lat_lons, centroid in zip(lat_lon_pairs, centroids):
            for lat, lon in lat_lons:
                to_write = [str(centroid.y), str(centroid.x), str(lat), str(lon)]                
                f.write(','.join(to_write) + '\n')

    print('Generated image download locations and saved at {}'.format(os.path.join(DATA_PROCESSED, 'image_download_locs.txt')))

if __name__ == '__main__':

    country = 'MWI'

    generate_grid(country)
    generate_country_download_locations(country)
