"""
Download images using the Planet API (and with a little rework, Google's; see ReadMe)

Written by Jatin Mathur.
5/2020
"""

import math
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from IPython.display import display
import matplotlib.pyplot as plt
import logging

BASE_DIR = '.'
import sys
sys.path.append(BASE_DIR)
from utils import create_space, PlanetDownloader
from config import RANDOM_SEED

COUNTRIES_DIR = os.path.join(BASE_DIR, 'data', 'countries')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
# can try using the google downloader, in which case change this to google_api_key.txt
ACCESS_TOKEN_DIR = os.path.join(BASE_DIR, 'planet_api_key.txt')

IMAGES_PER_CLUSTER = 20

def generate_download_locations(df, ipc=50):
    '''
    Takes a dataframe with columns cluster_lat, cluster_lon
    Generates a 10km x 10km bounding box around the cluster and samples 
    ipc images per cluster. First samples in a grid fashion, then any 
    remaining points are randomly (uniformly) chosen
    '''
    np.random.seed(RANDOM_SEED) # for reproducability
    df_download = {'image_name': [], 'image_lat': [], 'image_lon': []}
    for c in df.columns:
        df_download[c] = []
    # side length of square for uniform distribution
    edge_num = math.floor(math.sqrt(ipc))
    for _, r in df.iterrows():
        min_lat, min_lon, max_lat, max_lon = create_space(r.cluster_lat, r.cluster_lon)
        lats = np.linspace(min_lat, max_lat, edge_num).tolist()
        lons = np.linspace(min_lon, max_lon, edge_num).tolist()

        # performs cartesian product
        uniform_points = np.transpose([np.tile(lats, len(lons)), np.repeat(lons, len(lats))])
        
        lats = uniform_points[:,0].tolist()
        lons = uniform_points[:,1].tolist()
        
        # fills the remainder with random points
        for _ in range(ipc - edge_num * edge_num):
            lat = np.random.uniform(min_lat, max_lat)
            lon = np.random.uniform(min_lon, max_lon)
            lats.append(lat)
            lons.append(lon)
        
        # add to dict
        for lat, lon in zip(lats, lons):
            # image name is going to be image_lat_image_lon_cluster_lat_cluster_lon.png
            image_name = str(lat) + '_' + str(lon) + '_' + str(r.cluster_lat) + '_' + str(r.cluster_lon) + '.png'
            df_download['image_name'].append(image_name)
            df_download['image_lat'].append(lat)
            df_download['image_lon'].append(lon)
            for c in df.columns:
                df_download[c].append(r[c])
        
    return pd.DataFrame.from_dict(df_download)

def preprocess():
    if os.path.exists(os.path.join(PROCESSED_DIR, 'image_download_locs.csv')):
        print('Preprocessing already done')
        df_download = pd.read_csv(os.path.join(PROCESSED_DIR, 'image_download_locs.csv'))
        return df_download
    df_mw = pd.read_csv(os.path.join(COUNTRIES_DIR, 'malawi_2016', 'processed/clusters.csv'))
    df_eth = pd.read_csv(os.path.join(COUNTRIES_DIR, 'ethiopia_2015', 'processed/clusters.csv'))
    # for country in ['malawi_2016', 'ethiopia_2015']:
    #     os.makedirs(os.path.join(COUNTRIES_DIR, country, 'cnn_images'), exist_ok=False)
    df_mw_download = generate_download_locations(df_mw, IMAGES_PER_CLUSTER)
    print("malawi download size:", len(df_mw_download))
    df_eth_download = generate_download_locations(df_eth, IMAGES_PER_CLUSTER)
    print("ethiopia download size:", len(df_eth_download))
    df_download = pd.concat([df_mw_download, df_eth_download], axis=0)
    df_download.reset_index(drop=True, inplace=True)
    print("total download size:", len(df_download))
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df_download.to_csv(os.path.join(PROCESSED_DIR, 'image_download_locs.csv'), index=False)
    return df_download

def download_images(df):
    """
    Download images using a pandas DataFrame that has "image_lat", "image_lon", "image_name", "country" as columns
    
    Saves images to the corresponding inside COUNTRIES_DIR/<country>/cnn_images
    """
    access = None
    
    imd = PlanetDownloader(access)
    num_retries = 20
    wait_time = 0.1 # seconds

    # drops what is already downloaded
    already_downloaded = os.listdir(os.path.join(COUNTRIES_DIR, 'malawi_2016', 'cnn_images')) + \
                        os.listdir(os.path.join(COUNTRIES_DIR, 'ethiopia_2015', 'cnn_images'))
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
        country_dir = r.country
        image_save_path = os.path.join(COUNTRIES_DIR, country_dir, 'cnn_images', r.image_name)
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
                    print(f'Could not download {lat}, {lon} despite several retries and waiting')
                    continue
                else:
                    pass
            else:
                # no issues, save according to naming convention
                plt.imsave(image_save_path, im)

        except Exception as e:
            logging.error(f"Error-could not download {lat}, {lon}", exc_info=True)
            continue
    return

if __name__ == '__main__':
    df_download = preprocess()
    # download_images(df_download)
