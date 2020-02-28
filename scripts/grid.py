"""
Create 10km x 10km grid.

Written by Ed Oughton.

Winter 2020

"""
import argparse
import os
import configparser
import geopandas as gpd
from shapely.geometry import Polygon, mapping
import pandas as pd
import numpy as np
import rasterio
from rasterstats import zonal_stats

CONFIG = configparser.ConfigParser()
CONFIG.read('script_config.ini')

COUNTRY = CONFIG['DEFAULT']['COUNTRY']
SHAPEFILE_DIR = f'countries/{COUNTRY}/shapefile'
GRID_DIR = f'countries/{COUNTRY}/grid'


def create_folders():
    """
    Function to create new folder.

    """
    os.makedirs(GRID_DIR, exist_ok=True)


def generate_grid(country):
    """
    Generate a 10x10km spatial grid for the chosen country.

    """
    filename = 'national_outline_{}.shp'.format(country)
    country_outline = gpd.read_file(os.path.join(SHAPEFILE_DIR, filename))

    country_outline.crs = "epsg:4326"
    country_outline = country_outline.to_crs("epsg:3857")

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

    grid = gpd.GeoDataFrame({'geometry': polygons})
    intersection = gpd.overlay(grid, country_outline, how='intersection')
    intersection.crs = "epsg:3857"
    intersection = intersection.to_crs("epsg:4326")

    final_grid = query_settlement_layer(intersection)

    final_grid = final_grid[final_grid.geometry.notnull()]
    final_grid.to_file(os.path.join(GRID_DIR, 'grid.shp'))

    print('Completed grid generation process')


def query_settlement_layer(grid):
    """
    Query the settlement layer to get an estimated population for each grid square.

    """
    path = os.path.join(SHAPEFILE_DIR, f'{COUNTRY}.tif')

    grid['population'] = pd.DataFrame(
        zonal_stats(vectors=grid['geometry'], raster=path, stats='sum'))['sum']

    grid = grid.replace([np.inf, -np.inf], np.nan)

    return grid


if __name__ == '__main__':

    create_folders()

    generate_grid(COUNTRY)
