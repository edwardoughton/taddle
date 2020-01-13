"""
Create plots of grid-level predictions across a country

Written by Jatin Mathur

Winter 2020

"""

import configparser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry.point import Point
import matplotlib.cm as cm
import matplotlib.colors as colors

import sys
sys.path.append('.')
from utils import merge_on_lat_lon

import warnings
warnings.filterwarnings('ignore')

CONFIG = configparser.ConfigParser()
CONFIG.read('script_config.ini')

COUNTRY = CONFIG['DEFAULT']['COUNTRY']

def create_folders():
    os.makedirs(f'results/{COUNTRY}/figures/')

def create_plot(country, metric):
    print(f'creating plot for {metric}')
    df_geo = gpd.read_file(f'data/{country}/grid/grid.shp')
    df_geo['centroid'] = df_geo['geometry'].centroid
    df_geo['centroid_lat'] = df_geo['centroid'].apply(lambda point: point.y)
    df_geo['centroid_lon'] = df_geo['centroid'].apply(lambda point: point.x)
    preds = pd.read_csv(f'results/{country}/ridge_{metric}/predictions.csv')

    prev_len = len(df_geo)
    df_geo = merge_on_lat_lon(df_geo, preds, keys=['centroid_lat', 'centroid_lon'])
    assert len(df_geo) == prev_len, print('Merging geo-dataframe with predictions failed')

    geometry = df_geo['geometry']

    # if prediction is under 0, set to 0
    coloring_guide = df_geo[f'predicted_{metric}_pc']
    coloring_guide.loc[coloring_guide < 0] = 0

    cmap = 'inferno'
    vmin = coloring_guide.min()
    vmax = coloring_guide.max()

    kwargs = {'vmin': vmin,
            'vmax': vmax,
            'cmap': cmap}

    fig, ax = plt.subplots(figsize=(10,20))
    ax.set_aspect("equal")
    norm = colors.Normalize(vmin, vmax)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    gpd.plotting.plot_polygon_collection(ax, geometry, values=coloring_guide, **kwargs)

    units = ''
    if metric in ['consumption', 'phone_consumption']:
        units = '($/year)'

    label = (metric +' per capita').replace('_', ' ')
    ax.set_title(f'Malawi Predicted {label.title() + units}', fontsize=18)

    save_dir = f'results/{country}/figures/predicted_{metric}_per_capita.png'
    print(f'Saving figure to {save_dir}')
    plt.savefig(save_dir)

if __name__ == '__main__':

    arg = 'all'
    if len(sys.argv) >= 2:
        arg = sys.argv[1]
        assert arg in ['consumption', 'phone-consumption', 'phone-density']
        
    if arg == 'all':
        for metric in ['consumption', 'phone_consumption', 'phone_density']:
            create_plot(COUNTRY, metric)
    elif arg == 'consumption':
        create_plot(COUNTRY, 'consumption')
    elif arg == 'phone-consumption':
        create_plot(COUNTRY, 'phone_consumption')
    elif arg == 'phone-density':
        create_plot(COUNTRY, 'phone_density')
    else:
        raise ValueError('Args not handled correctly')


