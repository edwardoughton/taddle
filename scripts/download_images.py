import os
import configparser
import pandas as pd
import numpy as np
import random
import geopandas as gpd
from shapely.geometry import Point
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import logging

CONFIG = configparser.ConfigParser()
CONFIG.read('script_config.ini')

COUNTRY = CONFIG['DEFAULT']['COUNTRY']
GRID_DIR = CONFIG['DEFAULT']['GRID_DIR']
IMAGE_DIR = CONFIG['DEFAULT']['IMAGE_DIR']

ACCESS_TOKEN = None
with open(CONFIG['DEFAULT']['ACCESS_TOKEN_DIR'], 'r') as f:
    ACCESS_TOKEN = f.readlines()[0]


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


def generate_country_download_locations(country, num_per_grid=20):
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
    return pd.read_csv(os.path.join(GRID_DIR, 'image_download_locs.csv'))


class ImageryDownloader:
    def __init__(self, access_token):
        self.access_token = access_token
        self.url = 'https://maps.googleapis.com/maps/api/staticmap?center={},{}&zoom={}&size=400x400&maptype=satellite&key={}'
    
    def download(self, lat, long, zoom):
        res = requests.get(self.url.format(lat, long, zoom, self.access_token))
        image = Image.open(BytesIO(res.content))
        return image


def download_images(df):
    """
        Download images using a pandas DataFrame that has "image_lat", "image_lon", "image_name" as columns
    """
    imd = ImageryDownloader(ACCESS_TOKEN)
    zoom = 16

    # only download unique images
    df = df.dropna(subset=['image_lat', 'image_lon', 'image_name']).drop_duplicates(subset=['image_lat', 'image_lon'])
    # drops what is already downloaded
    already_downloaded = os.listdir(IMAGE_DIR)
    print('Already downloaded ' + str(len(already_downloaded)))
    df = df.set_index('image_name').drop(already_downloaded).reset_index()
    print('Need to download ' + str(len(df)))
    return

    for _, r in tqdm(df.iterrows(), total=df.shape[0]):
        lat = r.image_lat
        lon = r.image_lon
        name = r.image_name
        try:
            im = imd.download(lat, lon, zoom)
            # naming convention
            im.save(os.path.join(IMAGE_DIR, name))

        except Exception as e:
            logging.error("Error", exc_info=True)
            break



if __name__ == '__main__':
    print('Generating download locations...')
    df_download = generate_country_download_locations(COUNTRY)

    print('Downloading images. Might take a while...')
    download_images(df_download)
            