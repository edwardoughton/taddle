"""
Use baseline models using nightlights data and population data
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import geoio
from shapely.geometry import Polygon
from rasterstats import zonal_stats

BASE_DIR = '.'
import sys
sys.path.append(BASE_DIR)
from utils import merge_on_lat_lon, run_randomized_cv, run_spatial_cv, assign_groups, RidgeEnsemble, create_space
from config import TRAINING_CONFIG, RANDOM_SEED

COUNTRIES_DIR = os.path.join(BASE_DIR, 'data', 'countries')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

TYPE = TRAINING_CONFIG['TYPE']
COUNTRY = TRAINING_CONFIG['COUNTRY']
METRIC = TRAINING_CONFIG['METRIC']

# NIGHTLIGHTS_DIRS = [os.path.join(BASE_DIR, 'data/nightlights/viirs_2015_00N060W.tif'),
#                     os.path.join(BASE_DIR, 'data/nightlights/viirs_2015_75N060W.tif')]
# TIFS = [geoio.GeoImage(ndir) for ndir in NIGHTLIGHTS_DIRS]

assert TYPE in ['single_country', 'country_held_out']
assert COUNTRY in ['malawi_2016', 'ethiopia_2015']
assert METRIC in ['house_has_cellphone', 'est_monthly_phone_cost_pc']


def add_nightlights(df):
    ''' 
    This takes a dataframe with columns cluster_lat, cluster_lon and finds the average 
    nightlights in 2015 using a 10km x 10km box around the point
    '''
    return
    tif_array = None
    print('loading tif...')
    if COUNTRY == 'malawi_2016':
        print('loading tif...')
        tif_array = np.squeeze(tifs[0].get_data())
        add_nightlights(df_c, TIFS[0], tif_array)
    elif COUNTRY == 'ethiopia_2015':
        print('loading tif...')
        tif_array = np.squeeze(tifs[1].get_data())
        add_nightlights(df_c, TIFS[1], tif_array)
    else:
        raise ValueError('Unrecognized country')

    cluster_nightlights = []
    for i,r in df.iterrows():
        min_lat, min_lon, max_lat, max_lon = create_space(r.cluster_lat, r.cluster_lon)
        
        xminPixel, ymaxPixel = tif.proj_to_raster(min_lon, min_lat)
        xmaxPixel, yminPixel = tif.proj_to_raster(max_lon, max_lat)
        assert xminPixel < xmaxPixel, print(r.cluster_lat, r.cluster_lon)
        assert yminPixel < ymaxPixel, print(r.cluster_lat, r.cluster_lon)
        if xminPixel < 0 or xmaxPixel >= tif_array.shape[1]:
            print(f"no match for {r.cluster_lat}, {r.cluster_lon}")
            raise ValueError()
        elif yminPixel < 0 or ymaxPixel >= tif_array.shape[0]:
            print(f"no match for {r.cluster_lat}, {r.cluster_lon}")
            raise ValueError()
        xminPixel, yminPixel, xmaxPixel, ymaxPixel = int(xminPixel), int(yminPixel), int(xmaxPixel), int(ymaxPixel)
        cluster_nightlights.append(tif_array[yminPixel:ymaxPixel,xminPixel:xmaxPixel].mean())
        
    df['nightlights'] = cluster_nightlights
    return


def create_poly(r):
    lat = r.cluster_lat
    lon = r.cluster_lon
    min_lat, min_lon, max_lat, max_lon = create_space(lat, lon)
    points = [(min_lon, min_lat), (min_lon, max_lat), (max_lon, max_lat), (max_lon, max_lat)]
    return Polygon(points)


def add_population_data(df):
    """
    This takes a dataframe with columns cluster_lat, cluster_lon and finds the total 
    population in 2015 using a 10km x 10km box around the point
    """
    country_abbrv = None
    if COUNTRY == 'malawi_2016':
        country_abbrv = 'MWI'
    elif COUNTRY == 'ethiopia_2015':
        country_abbrv = 'ETH'
    else:
        raise ValueError("unrecognized country")
    shapefile_dir = os.path.join(COUNTRIES_DIR, country_abbrv, 'shapefile')
    path = os.path.join(shapefile_dir, f'{country_abbrv}.tif')
    if not os.path.exists(path):
        print("you need to run extract_shapefile.py to generate the population tif for this country")
        raise ValueError()

    geometries = df.apply(create_poly, axis=1)
    df['population'] = pd.DataFrame(zonal_stats(vectors=geometries, raster=path, stats='sum'))['sum']
    df['population'] = df['population'].replace([np.inf, -np.inf], np.nan)
    print("population nulls:", df['population'].isna().sum())
    return


def load_data():
    df_c = pd.read_csv(os.path.join(PROCESSED_DIR, TYPE, COUNTRY, f'{METRIC}.csv'))
    df_c = df_c.groupby(['cluster_lat', 'cluster_lon']).mean()
    df_c = df_c.reset_index().drop(['image_lat', 'image_lon', 'bin', 'near_lower', 'near_upper'], axis=1)
    add_nightlights(df_c)
    add_population_data(df_c)
    df_train = df_c[df_c['is_train']].copy()
    df_valid = df_c[~df_c['is_train']].copy()
    return df_train, df_valid


if __name__ == '__main__':
    df_train, df_valid = load_data()
    for baseline in ['nightlights', 'population']:
        print()
        print('using baseline:', baseline)
        median = df_train[baseline].median()
        print('filling any nulls with the median')
        x_train = df_train[baseline].fillna(median).values.reshape(-1, 1)
        x_valid = df_valid[baseline].fillna(median).values.reshape(-1, 1)
        y_train = df_train[METRIC].values
        y_valid = df_valid[METRIC].values
        print("running randomized cv...")
        r2_rcv, _, ridges_rcv, scalers_rcv = run_randomized_cv(x_train, y_train, random_seed=RANDOM_SEED)
        re_rcv = RidgeEnsemble(ridges_rcv, scalers_rcv)
        yhat_rcv_valid = re_rcv.predict(x_valid)
        r2_rcv_valid = r2_score(y_valid, yhat_rcv_valid)
        print(f"randomized cv r2: {r2_rcv}, validation r2: {r2_rcv_valid}")
        pearson_r2_rcv = pearsonr(y_valid, yhat_rcv_valid)[0]**2
        print(f"validation pearson R squared: {pearson_r2_rcv}")

        print()
        
        groups, _ = assign_groups(df_train, 5, random_seed=RANDOM_SEED)
        print("running spatial cv...")
        r2_scv, yhat_scv_train, ridges_scv, scalers_scv = run_spatial_cv(x_train, y_train, groups, random_seed=RANDOM_SEED)
        re_scv = RidgeEnsemble(ridges_scv, scalers_scv)
        yhat_scv_valid = re_scv.predict(x_valid)
        r2_scv_valid = r2_score(y_valid, yhat_scv_valid)
        print(f"spatial cv r2: {r2_scv}, validation r2: {r2_scv_valid}")
        pearson_r2_scv = pearsonr(y_valid, yhat_scv_valid)[0]**2
        print(f"validation pearson R squared: {pearson_r2_scv}")

        print()

