"""
Create 10km x 10km grid.

Written by Ed Oughton

Winter 2020

"""
import argparse
import os
import configparser
import geopandas as gpd
from shapely.geometry import Polygon
import pandas as pd
import numpy as np

CONFIG = configparser.ConfigParser()
CONFIG.read('script_config.ini')

COUNTRY = CONFIG['DEFAULT']['COUNTRY']
SHAPEFILE_DIR = f'data/{COUNTRY}/shapefile'
GRID_DIR = f'data/{COUNTRY}/grid'
IMAGE_DIR = f'data/{COUNTRY}/images'

def create_folders():
    os.makedirs(GRID_DIR, exist_ok=True)

def generate_grid(country):

    filename = 'national_outline_{}.shp'.format(country)
    country_outline = gpd.read_file(os.path.join(SHAPEFILE_DIR, filename))

    country_outline.crs = {'init':'epsg:4326'}
    country_outline = country_outline.to_crs({'init':'epsg:3857'})

    xmin,ymin,xmax,ymax = country_outline.total_bounds

    #10km sides, leading to 100km^2 area
    length = 1e4
    wide = 1e4

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

    intersection.to_file(os.path.join(GRID_DIR, 'grid.shp'))

    print('Completed grid generation process')


if __name__ == '__main__':
    create_folders()
    generate_grid(COUNTRY)
