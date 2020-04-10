"""
Visualization script

Written by Jatin Mathur and Ed Oughton.

January 2020

"""
import os
import configparser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import matplotlib.colors
import geoio
import math
import warnings
from sklearn.metrics import r2_score
warnings.filterwarnings('ignore')

CONFIG_DATA = configparser.ConfigParser()
CONFIG_DATA.read(os.path.join(os.path.dirname(__file__), '..', 'scripts','script_config.ini'))

CONFIG_COUNTRY = configparser.ConfigParser()
CONFIG_COUNTRY.read('script_config.ini')
COUNTRY = CONFIG_COUNTRY['DEFAULT']['COUNTRY']
SHAPEFILE_DIR = f'countries/{COUNTRY}/shapefile'
GRID_DIR = f'countries/{COUNTRY}/grid'
RESULTS_DIR = f'countries/{COUNTRY}/results'
WORLDPOP = 'data/world_population'
CLUSTER_DATA_DIR = f'data/LSMS/{COUNTRY}/processed/cluster_data.csv'
CLUSTER_FIGURES_DIR = f'data/LSMS/{COUNTRY}/figures'
CLUSTER_PREDICTIONS_DIR = f'data/LSMS/{COUNTRY}/output/cluster_predictions.csv'

# Purchasing Power Adjustment
PPP = float(CONFIG_COUNTRY['DEFAULT']['PPP'])

def create_folders():
    os.makedirs(CLUSTER_FIGURES_DIR, exist_ok=True)

def prepare_data():
    """
    Preprocessing function.

    """

    print("Preprocessing...")

    df_clusters = pd.read_csv(CLUSTER_PREDICTIONS_DIR)

    filename = 'ppp_2020_1km_Aggregated.tif'
    img = geoio.GeoImage(os.path.join(WORLDPOP, filename))
    im_array = np.squeeze(img.get_data())

    cluster_population = []
    for _, r in df_clusters.iterrows():
        min_lat, min_lon, max_lat, max_lon = create_space(r.cluster_lat, r.cluster_lon)
        xminPixel, yminPixel = img.proj_to_raster(min_lon, min_lat)
        xmaxPixel, ymaxPixel = img.proj_to_raster(max_lon, max_lat)

        xminPixel, xmaxPixel = min(xminPixel, xmaxPixel), max(xminPixel, xmaxPixel)
        yminPixel, ymaxPixel = min(yminPixel, ymaxPixel), max(yminPixel, ymaxPixel)

        xminPixel, yminPixel, xmaxPixel, ymaxPixel = (
            int(xminPixel), int(yminPixel), int(xmaxPixel), int(ymaxPixel)
        )
        cluster_population.append(
            round(im_array[yminPixel:ymaxPixel,xminPixel:xmaxPixel].mean()))

    df_clusters['cluster_population_density_1km2'] = cluster_population

    return df_clusters


def create_space(lat, lon):
    """
    Creates a 100km^2 area bounding box.

    Parameters
    ----------
    lat : float
        Latitude
    lon : float
        Longitude

    """
    v = (180 / math.pi) * (5000 / 6378137) # approximately 0.045
    min_lat = lat - v
    min_lon = lon - v
    max_lat = lat + v
    max_lon = lon + v

    return min_lat, min_lon, max_lat, max_lon


def corr_coef(x, y):

    coef = round(np.corrcoef(x, y)[0, 1], 2)

    return coef


def create_regplot(data):

    print("Creating regplot")

    data = data[data['cluster_annual_consumption_pc'] <= 10000]

    data['cluster_monthly_phone_consumption_pc'] = data['cluster_annual_phone_consumption_pc'] / 12

    bins = [0, 400, 800, 1200, 4020]
    labels = [5, 15, 25, 60]
    data['pop_density_binned'] = pd.cut(data['cluster_population_density_1km2'], bins=bins, labels=labels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    coef1 = corr_coef(data['cluster_nightlights'], data['cluster_cellphones_pc'])
    g = sns.regplot(y="cluster_nightlights", x="cluster_cellphones_pc",
        data=data, ax=ax1,
        scatter_kws={'alpha': 0.2, 'color':'blue'},
        line_kws={'alpha': 0.5, 'color':'black'}
        )
    g.set(ylabel='Luminosity (DN)', xlabel='Cell Phones Per Capita',
        title='Luminosity vs\nCell Phones Per Capita (R={})'.format(str(coef1)),
        xlim=(0, 1), ylim=(0, 50))

    coef2 = corr_coef(data['cluster_nightlights'], data['cluster_monthly_phone_consumption_pc'])
    g = sns.regplot(y="cluster_nightlights", x="cluster_monthly_phone_consumption_pc",
        data=data, ax=ax2, scatter_kws={'alpha': 0.2, 'color':'blue'},
        line_kws={'alpha': 0.5, 'color':'black'})
    g.set(ylabel='Luminosity (DN)', xlabel='Consumption ($ per month)',
        title='Luminosity vs\nMonthly Phone Consumption (R={})'.format(str(coef2)),
         xlim=(0, 30), ylim=(0, 50))


    fig.tight_layout()
    save_dir = os.path.join(CLUSTER_FIGURES_DIR, 'regplot_luminosity.png')
    print(f'Saving figure to {save_dir}')
    fig.savefig(save_dir)

    print("Plotting completed")


def create_prediction_regplot(data):

    print("Creating regplot")

    # 'cluster_annual_phone_consumption_pc',*
    #    'cluster_cellphones_pc', *

    data = data[data['cluster_annual_consumption_pc'] <= 10000]

    # data['cluster_monthly_consumption_pc'] = data['cluster_annual_consumption_pc'] / 12
    data['cluster_monthly_phone_consumption_pc'] = data['cluster_annual_phone_consumption_pc'] / 12
    # data['cluster_monthly_phone_cost_pc'] = data['cluster_estimated_annual_phone_cost_pc'] / 12
    # data['predicted_monthly_consumption'] = data['predicted_consumption'] / 12
    data['predicted_monthly_phone_consumption'] = data['predicted_phone_consumption'] / 12

    bins = [0, 400, 800, 1200, 4020]
    labels = [5, 15, 25, 60]
    data['pop_density_binned'] = pd.cut(data['cluster_population_density_1km2'], bins=bins, labels=labels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    coef1 = round(r2_score(data['cluster_cellphones_pc'], data['predicted_phone_density']), 2)
    #Check: I'm not sure 'predicted_phone_density' is a density - isn't it penetration rate?
    g = sns.regplot(x="cluster_cellphones_pc", y="predicted_phone_density",
        data=data, ax=ax1, scatter_kws={'alpha': 0.2, 'color':'blue'},
        line_kws={'alpha': 0.5, 'color':'black'})
    g.set(xlabel='Observed ($)', ylabel='Predicted ($)',
        title='Cell Phones Per Capita:\nObserved vs Predicted (R$^2$={})'.format(str(coef1)),
        xlim=(0, 0.6), ylim=(0, 0.6))

    coef2 = round(r2_score(data['cluster_monthly_phone_consumption_pc'], data['predicted_monthly_phone_consumption']), 2)
    g = sns.regplot(x="cluster_monthly_phone_consumption_pc", y="predicted_monthly_phone_consumption",
        data=data, ax=ax2, scatter_kws={'alpha': 0.2, 'color':'blue'},
        line_kws={'alpha': 0.5, 'color':'black'})
    g.set(xlabel='Observed ($)', ylabel='Predicted ($)', #predicted phone cost
        title='Monthly Phone Consumption:\nObserved vs Predicted (R$^2$={})'.format(str(coef2)),
        xlim=(0, 10), ylim=(0, 10))

    fig.tight_layout()
    save_dir = os.path.join(CLUSTER_FIGURES_DIR, 'regplot_predictions.png')
    print(f'Saving figure to {save_dir}')
    fig.savefig(save_dir)

    print("Plotting completed")


if __name__ == '__main__':

    create_folders()

    data = prepare_data()

    # Index(['cluster_lat', 'cluster_lon', 'cluster_persons_surveyed',
    #    'cluster_annual_consumption_pc',
    # 'cluster_annual_phone_consumption_pc',*
    #    'cluster_cellphones_pc', *
    # 'cluster_estimated_annual_phone_cost_pc',
    #    'cluster_nightlights', 'predicted_consumption',
    #    'predicted_log_consumption', 'predicted_phone_consumption',
    #    'predicted_log_phone_consumption', 'predicted_phone_density',
    #    'predicted_log_phone_density', 'cluster_population_density_1km2'],
    #   dtype='object')

    create_regplot(data)

    create_prediction_regplot(data)
