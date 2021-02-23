"""
Create plots of grid-level predictions across a country

Written by Jatin Mathur and Ed Oughton.

Winter 2020

"""
import os
import sys
import configparser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry.point import Point
import matplotlib.cm as cm
import matplotlib.colors as colors
import contextily as ctx
sys.path.append('.')
from utils import merge_on_lat_lon

import warnings
warnings.filterwarnings('ignore')

CONFIG = configparser.ConfigParser()
CONFIG.read('script_config.ini')

COUNTRY = CONFIG['DEFAULT']['COUNTRY']
GRID_DIR = f'countries/{COUNTRY}/grid'
RESULTS_DIR = f'countries/{COUNTRY}/results/'


def create_folders():
    """
    Function to create desired folder.

    """
    os.makedirs(os.path.join(RESULTS_DIR, 'figures'), exist_ok=True)


def create_plot(country, metric, min_population=1000, under_color='#696969'):
    """
    This method creates a geospatial figure depicting the predictions
    for a given metric within a grid.

    Currently, the color scale terminates 3 standard deviations above
    and below the mean. This is to prevent outliers with extremely high
    relative consumptions from dominating the linear color scale.

    Parameters
    ----------
    country : string
        Country to plot (ISO 3 digit code).
    metric : string
        Metric to plot (consumption, phone_consumption, phone_density).
    min_population : int
        The minimum population a grid should have to included in
        the scale `under_color`. And squares with populations under
        the `min_population` are filled based on the color selected
        in `under_color`.
    under_color : string
        Desired color for areas which don't exceed `min_population`.

    """
    print(f'creating plot for {metric}')
    df_geo = gpd.read_file(os.path.join(GRID_DIR, 'grid.shp'))
    df_geo['centroid'] = df_geo['geometry'].centroid
    df_geo['centroid_lat'] = df_geo['centroid'].apply(lambda point: point.y)
    df_geo['centroid_lon'] = df_geo['centroid'].apply(lambda point: point.x)
    preds = pd.read_csv(os.path.join(RESULTS_DIR, f'ridge_{metric}', 'predictions.csv'))

    if min_population is None:
        df_geo['to_ignore'] = False
    else:
        to_use = df_geo['population'] >= min_population
        df_geo['to_ignore'] = True
        df_geo.loc[to_use, 'to_ignore'] = False

    df_geo = merge_on_lat_lon(df_geo, preds, keys=['centroid_lat', 'centroid_lon'], how='left')

    geometry = df_geo['geometry']

    # if prediction is under 0, set to 0
    coloring_guide = df_geo[f'predicted_{metric}_pc']
    if metric in ['consumption', 'phone_consumption']:
        coloring_guide /= 12 # we want to show monthly
    coloring_guide.loc[coloring_guide < 0] = 0
    coloring_guide.fillna(-1, inplace=True)
    coloring_guide.loc[df_geo['to_ignore']] = -1

    vmin = coloring_guide.mean() - 2 * coloring_guide.std()
    if vmin < 0:
        vmin = 0
    elif vmin < coloring_guide.min():
        vmin = coloring_guide.min()
    vmax = coloring_guide.mean() + 2 * coloring_guide.std()
    if vmax > coloring_guide.max():
        vmax = coloring_guide.max()

    cmap = cm.get_cmap('inferno')
    cmap.set_under(under_color)

    kwargs = {
        'vmin': vmin,
        'vmax': vmax,
        'cmap': cmap
    }

    fig, ax = plt.subplots(figsize=(6,12))

    ax.set_aspect("equal")
    norm = colors.Normalize(vmin, vmax)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    gpd.plotting.plot_polygon_collection(ax, geometry, values=coloring_guide, **kwargs)

    units = ''
    if metric in ['consumption', 'phone_consumption']:
        units = '($/month)'

    label = (metric +' per capita').replace('_', ' ')
    ax.set_title(f'Malawi Predicted {label.title() + units}\n (min pop per grid: {min_population})', fontsize=10)
    ctx.add_basemap(ax, crs=df_geo.crs)

    save_dir = os.path.join(RESULTS_DIR, 'figures', f'predicted_{metric}_per_capita.png')
    print(f'Saving figure to {save_dir}')
    plt.savefig(save_dir)

    print('Plotting completed')


if __name__ == '__main__':

    create_folders()

    arg = '--all'
    if len(sys.argv) >= 2:
        arg = sys.argv[1]
        assert arg in ['--consumption', '--phone-consumption', '--phone-density']

    if arg == '--all':
        for metric in ['consumption', 'phone_consumption', 'phone_density']:
            create_plot(COUNTRY, metric)
    elif arg == '--consumption':
        create_plot(COUNTRY, 'consumption')
    elif arg == '--phone-consumption':
        create_plot(COUNTRY, 'phone_consumption')
    elif arg == '--phone-density':
        create_plot(COUNTRY, 'phone_density')
    else:
        raise ValueError('Args not handled correctly')
