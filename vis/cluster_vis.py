"""
Visualization script

Written by Jatin Mathur and Ed Oughton.

Winter 2020

Written by Jatin Mathur.
5/2020
"""
import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import matplotlib.colors
import geoio
import math
import warnings
warnings.filterwarnings('ignore')

import sys
# wherever we are called from, we move to the directory of this script
self_dir = os.path.dirname(os.path.join(os.getcwd(), sys.argv[0]))
os.chdir(self_dir)

BASE_DIR = ".."
sys.path.append(BASE_DIR)
from config import VIS_CONFIG

COUNTRY_NAME = VIS_CONFIG['COUNTRY_NAME']
TYPE = VIS_CONFIG['TYPE']
COUNTRY = VIS_CONFIG['COUNTRY']
METRIC = VIS_CONFIG['METRIC']

RESULTS_DIR = os.path.join(BASE_DIR, f'results/{TYPE}/{COUNTRY}/')
WORLDPOP = os.path.join(BASE_DIR, 'data/world_population')
CLUSTER_PREDICTIONS_DIR = os.path.join(BASE_DIR, f'data/LSMS/{COUNTRY}/output/cluster_predictions.csv')
CLUSTER_FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

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

        arr = im_array[yminPixel:ymaxPixel,xminPixel:xmaxPixel]
        arr[ arr < 0 ] = 0 # can't have negative populations
        cluster_population.append(
            round(arr.mean())
        )

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


def r2(x, y):
    coef = round(np.corrcoef(x, y)[0, 1]**2, 3)
    return coef


def load_predictions():
    return pd.read_csv(os.path.join(RESULTS_DIR, METRIC, 'cluster_predictions', f'{METRIC}.csv'))


def solve_std(y_train, y_hat_train):
    # probabilistic linear regression has this solution for the optimal standard deviation after finding the predictions
    # see paper for assumptions made
    std = np.sqrt(((y_hat_train - y_train)**2).sum() / len(y_train))
    return std


def plot_predictions():
    df_preds = load_predictions()
    df_train = df_preds[df_preds['is_train']]
    std = solve_std(df_train[METRIC].values, df_train[f'pred_{METRIC}'].values)
    df_valid = df_preds[~df_preds['is_train']]
    y = df_valid[METRIC].values
    yhat = df_valid[f'pred_{METRIC}'].values
    r2 = r2_score(y, yhat)
    # for the purposes of clean plotting, we won't show anything too far beyond 
    # the max of what our model predicts
    max_y = None if METRIC == 'house_has_cellphone' else max(yhat) + 2
    if max_y is not None:
        yhat = yhat[y < max_y]
        y = y[y < max_y]
    # these are ordered for the plotting
    y_i = np.unique(y)
    yhat_i = np.poly1d(np.polyfit(y, yhat, 1))(np.unique(y))
    pi = 1 * std # prediction interval

    fig, ax = plt.subplots()
    ax.scatter(y, yhat, alpha=0.4)
    ax.plot(y_i, yhat_i)
    xloc = 0.8 * max(y_i)
    yloc = 0.75 * max(yhat_i)
    plt.text(xloc, yloc, f'r^2={round(r2, 3)}', size=12)
    ax.fill_between(y_i, (yhat_i - pi), (yhat_i + pi), color='b', alpha=.1)

    title_label = None
    x_label = None
    y_label = None
    if METRIC == 'house_has_cellphone':
        title_label = 'Device Penetration'
        x_label = "Observed Devices Per Capita"
        y_label = "Predicted Devices Per Capita"
    elif METRIC == 'est_monthly_phone_cost_pc':
        title_label = 'Spend on Phone Services Per Capita ($/mo)'
        x_label = "Observed Spend ($/mo)"
        y_label = "Predicted Spend ($/mo)"
    else:
        raise ValueError(f"Unsupported metric {METRIC}")
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f'{COUNTRY_NAME} {title_label}: \nObserved vs Predicted with 1 Stdv. Prediction Intervals')

    savepath = os.path.join(CLUSTER_FIGURES_DIR, f'{METRIC}.png')
    print(f'saving to {savepath}')
    fig.savefig(savepath)


if __name__ == '__main__':
    create_folders()
    plot_predictions()
