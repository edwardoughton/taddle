"""
Generate download locations within a country and download them.

Written by Jatin Mathur.
5/2020
"""
import os
import configparser
import math
import pandas as pd
import numpy as np
import random
import geopandas as gpd
from shapely.geometry import Point
import requests
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import logging
import time

BASE_DIR = '.'
# repo imports
import sys
sys.path.append(BASE_DIR)
from utils import PlanetDownloader
from config import VIS_CONFIG, RANDOM_SEED

COUNTRY_ABBRV = VIS_CONFIG['COUNTRY_ABBRV']
COUNTRIES_DIR = os.path.join(BASE_DIR, 'data', 'countries')
GRID_DIR = os.path.join(COUNTRIES_DIR, COUNTRY_ABBRV, 'grid')
IMAGE_DIR = os.path.join(COUNTRIES_DIR, COUNTRY_ABBRV, 'images')

ACCESS_TOKEN_DIR = os.path.join(BASE_DIR, 'planet_api_key.txt')
ACCESS_TOKEN = None
with open(ACCESS_TOKEN_DIR, 'r') as f:
    ACCESS_TOKEN = f.readlines()[0]
assert ACCESS_TOKEN is not None, print("Access token is not valid")


def create_folders():
    """
    Function to create new folders.
    """
    os.makedirs(IMAGE_DIR, exist_ok=True)


def get_polygon_download_locations(polygon, number, seed=7):
    """
    Samples NUMBER points evenly but randomly from a polygon. Seed is set for
    reproducibility.

    At first tries to create sub-grid of size n x n where n = sqrt(number)
    It checks these coordinates and if they are in the polygon it uses them

    If the number of points found is still less than the desired number,
    samples are taken randomly from the polygon until the required number
    is achieved.
    """
    random.seed(seed)

    min_x, min_y, max_x, max_y = polygon.bounds
    edge_num = math.floor(math.sqrt(number))
    lats = np.linspace(min_y, max_y, edge_num)
    lons = np.linspace(min_x, max_x, edge_num)

    # performs cartesian product
    evenly_spaced_points = np.transpose(
        [np.tile(lats, len(lons)), np.repeat(lons, len(lats))])
    assert len(evenly_spaced_points) <= number

    # tries using evenly spaced points
    points = []
    for proposed_lat, proposed_lon in evenly_spaced_points:
        point = Point(proposed_lon, proposed_lat)
        if polygon.contains(point):
            points.append([proposed_lat, proposed_lon])

    # fills the remainder with random points
    while len(points) < number:

        point = Point(random.uniform(min_x, max_x),
            random.uniform(min_y, max_y))

        if polygon.contains(point):
            points.append([point.y, point.x])

    return points  # returns list of lat/lon pairs


def generate_country_download_locations(min_population=100, num_per_grid=4):
    """
    Generates a defined number of download locations (NUM_PER_GRID) for each
    grid with at least the minimum number of specified residents (MIN_
    POPULATION).
    """
    grid = gpd.read_file(os.path.join(GRID_DIR, 'grid.shp'))
    grid = grid[grid['population'] >= min_population]

    lat_lon_pairs = grid['geometry'].apply(
        lambda polygon: get_polygon_download_locations(
            polygon, number=num_per_grid))

    centroids = grid['geometry'].centroid

    columns = [
        'centroid_lat', 'centroid_lon', 'image_lat', 'image_lon', 'image_name'
    ]

    with open(os.path.join(GRID_DIR, 'image_download_locs.csv'), 'w') as f:
        f.write(','.join(columns) + '\n')

        for lat_lons, centroid in zip(lat_lon_pairs, centroids):
            for lat, lon in lat_lons:
                name = str(lat) + '_' + str(lon) + '.png'
                to_write = [
                    str(centroid.y), str(centroid.x), str(lat), str(lon), name]
                f.write(','.join(to_write) + '\n')

    print('Generated image download locations and saved at {}'.format(
        os.path.join(GRID_DIR, 'image_download_locs.csv')))


def download_images(df):
    """
    Download images using a pandas DataFrame that has "image_lat", "image_lon",
    "image_name" as columns.
    """
    imd = PlanetDownloader(ACCESS_TOKEN)
    zoom = 16
    NUM_RETRIES = 20
    WAIT_TIME = 0.1 # seconds

    # drops what is already downloaded
    already_downloaded = os.listdir(IMAGE_DIR)
    print('Already downloaded ' + str(len(already_downloaded)))

    df = df.set_index('image_name').drop(already_downloaded).reset_index()
    print('Need to download ' + str(len(df)))

    # use three years of images to find one that matches search critera
    min_year = 2014
    min_month = 1
    max_year = 2016
    max_month = 12

    for _, r in tqdm(df.iterrows(), total=df.shape[0]):
        lat = r.image_lat
        lon = r.image_lon
        name = r.image_name
        try:
            im = imd.download_image(lat, lon, min_year, min_month, max_year, max_month)
            if im is None:
                resolved = False
                for _ in range(num_retries):
                    time.sleep(wait_time)
                    im = imd.download_image(lat, lon, min_year, min_month, max_year, max_month)
                    if im is None:
                        continue
                    else:
                        plt.imsave(image_save_path, im)
                        resolved = True
                        break
                if not resolved:
                    # print(f'Could not download {lat}, {lon} despite several retries and waiting')
                    continue
                else:
                    pass
            else:
                # no issues, save according to naming convention
                plt.imsave(os.path.join(IMAGE_DIR, name), im)

        except Exception as e:
            # logging.error(f"Error-could not download {lat}, {lon}", exc_info=True)
            continue
    return

if __name__ == '__main__':

    create_folders()

    arg = '--all'
    if len(sys.argv) >= 2:
        arg = sys.argv[1]
        assert arg in ['--all', '--generate-download-locations', '--download-images']

    if arg == '--all':
        print('Generating download locations...')
        generate_country_download_locations()

        df_download = pd.read_csv(os.path.join(GRID_DIR, 'image_download_locs.csv'))

        print('Downloading images. Might take a while...')
        download_images(df_download)

    elif arg == '--generate-download-locations':
        print('Generating download locations...')
        generate_country_download_locations()

    elif arg == '--download-images':
        df_download = pd.read_csv(os.path.join(GRID_DIR, 'image_download_locs.csv'))

        print('Downloading images. Might take a while...')
        download_images(df_download)

    else:
        raise ValueError('Args not handled correctly')
