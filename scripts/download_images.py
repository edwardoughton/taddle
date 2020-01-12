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

# repo imports
import sys
sys.append('.')
from utils import ImageryDownloader

CONFIG = configparser.ConfigParser()
CONFIG.read('script_config.ini')

COUNTRY = CONFIG['DEFAULT']['COUNTRY']
GRID_DIR = CONFIG['DEFAULT']['GRID_DIR']
IMAGE_DIR = CONFIG['DEFAULT']['IMAGE_DIR']

ACCESS_TOKEN = None
with open(CONFIG['DEFAULT']['ACCESS_TOKEN_DIR'], 'r') as f:
    ACCESS_TOKEN = f.readlines()[0]

def create_folders():
    os.makedirs(IMAGE_DIR, exist_ok=True)

def get_polygon_download_locations(polygon, number, seed=7):
    """
        Samples NUMBER points evenly but randomly from a polygon
        Seed is set to 7 for reproducibility

        At first tries to create sub-grid of size n x n where n = sqrt(number)
        It checks these coordinates and if they are in the polygon it uses them

        If the number of points found is still less than the number, I sample randomly
        from the polygon until I have the required number
    """
    random.seed(seed)

    min_x, min_y, max_x, max_y = polygon.bounds
    edge_num = math.floor(math.sqrt(number))
    lats = np.linspace(min_y, max_y, edge_num)
    lons = np.linspace(min_x, max_x, edge_num)
    # performs cartesian product
    evenly_spaced_points = np.transpose([np.tile(lats, len(lons)), np.repeat(lons, len(lats))])
    assert len(evenly_spaced_points) <= number

    # tries using evenly spaced points
    points = []
    for proposed_lat, proposed_lon in evenly_spaced_points:
        point = Point(proposed_lon, proposed_lat)
        if polygon.contains(point):
            points.append([proposed_lat, proposed_lon])

    # fills the remainder with random points
    while len(points) < number:
        point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        if polygon.contains(point):
            points.append([point.y, point.x])

    return points  # returns list of lat/lon pairs


def generate_country_download_locations(country, num_per_grid=100):
    grid = gpd.read_file(os.path.join(GRID_DIR, 'grid.shp'))
    lat_lon_pairs = grid['geometry'].apply(lambda polygon: get_polygon_download_locations(polygon, number=num_per_grid))
    centroids = grid['geometry'].centroid

    columns = ['centroid_lat', 'centroid_lon', 'image_lat', 'image_lon', 'image_name']
    with open(os.path.join(GRID_DIR, 'image_download_locs.csv'), 'w') as f:
        f.write(','.join(columns) + '\n')

        for lat_lons, centroid in zip(lat_lon_pairs, centroids):
            for lat, lon in lat_lons:
                name = str(lat) + '_' + str(lon) + '.png'
                to_write = [str(centroid.y), str(centroid.x), str(lat), str(lon), name]                
                f.write(','.join(to_write) + '\n')

    print('Generated image download locations and saved at {}'.format(os.path.join(GRID_DIR, 'image_download_locs.csv')))


def download_images(df):
    """
        Download images using a pandas DataFrame that has "image_lat", "image_lon", "image_name" as columns
    """
    imd = ImageryDownloader(ACCESS_TOKEN)
    zoom = 16
    # BAD_IMAGE_VALUE = 0.8784313797950745
    NUM_RETRIES = 20
    WAIT_TIME = 0.1 # seconds

    # only download unique images
    df = df.dropna(subset=['image_lat', 'image_lon', 'image_name']).drop_duplicates(subset=['image_lat', 'image_lon'])
    # drops what is already downloaded
    already_downloaded = os.listdir(IMAGE_DIR)
    print('Already downloaded ' + str(len(already_downloaded)))
    df = df.set_index('image_name').drop(already_downloaded).reset_index()
    print('Need to download ' + str(len(df)))

    for _, r in tqdm(df.iterrows(), total=df.shape[0]):
        lat = r.image_lat
        lon = r.image_lon
        name = r.image_name
        try:
            im = imd.download(lat, lon, zoom)
            if type(im) == str and im == 'RETRY':
                resolved = False
                for _ in range(NUM_RETRIES):
                    time.sleep(WAIT_TIME)
                    im = imd.download(lat, lon, zoom)
                    if type(im) == str and im == 'RETRY':
                        continue
                    else:
                        plt.imsave(os.path.join(IMAGE_DIR, name), im)
                        resolved = True
                        break
                if not resolved:
                    raise ValueError(f'Could not download {lat}, {lon}, {zoom} despite several retries and waiting')
                else:
                    pass
            else:
                # no issues, save according to naming convention
                plt.imsave(os.path.join(IMAGE_DIR, name), im)

        except Exception as e:
            logging.error(f"Error-could not download {lat}, {lon}, {zoom}", exc_info=True)
            break



if __name__ == '__main__':
    create_folders()
    
#     print('Generating download locations...')
#     generate_country_download_locations(COUNTRY)
    
    df_download = pd.read_csv(os.path.join(GRID_DIR, 'image_download_locs.csv'))

    print('Downloading images. Might take a while...')
    download_images(df_download)
            