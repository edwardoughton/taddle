"""
Visualize grid-level predictions across a country

Written by Jatin Mathur
5/2020
"""
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry.point import Point
import matplotlib.cm as cm
import matplotlib.colors as colors
import contextily as ctx

BASE_DIR = '.'
import sys
sys.path.append(BASE_DIR)
from utils import merge_on_lat_lon
from config import PREDICTION_MAPS_CONFIG

COUNTRIES_DIR = os.path.join(BASE_DIR, 'data', 'countries')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

COUNTRY_ABBRV = PREDICTION_MAPS_CONFIG['COUNTRY_ABBRV']
CNN_GRID_OUTPUTS = os.path.join(RESULTS_DIR, 'prediction_maps', COUNTRY_ABBRV, 'cnn')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'prediction_maps', COUNTRY_ABBRV, 'figures')
GRID_DIR = os.path.join(COUNTRIES_DIR, COUNTRY_ABBRV, 'grid')

TYPE = PREDICTION_MAPS_CONFIG['TYPE']
COUNTRY = PREDICTION_MAPS_CONFIG['COUNTRY']
METRIC = PREDICTION_MAPS_CONFIG['METRIC']

assert TYPE in ['single_country', 'country_held_out']
assert COUNTRY in ['malawi_2016', 'ethiopia_2015']
assert METRIC in ['house_has_cellphone', 'est_monthly_phone_cost_pc']


def create_folders():
    os.makedirs((FIGURES_DIR), exist_ok=True)


def create_plot(min_population=100, under_color='gray'):
    """
    This method creates a geospatial figure depicting the predictions
    for a given metric within a grid.

    Currently, the color scale terminates 3 standard deviations above
    and below the mean. This is to prevent outliers with extremely high
    relative consumptions from dominating the linear color scale.

    Parameters
    ----------
    min_population : int
        The minimum population a grid should have to included in
        the scale `under_color`. And squares with populations under
        the `min_population` are filled based on the color selected
        in `under_color`.
    under_color : string
        Desired color for areas which don't exceed `min_population`.
    """
    print(f'creating plot for {METRIC}')
    df_geo = gpd.read_file(os.path.join(GRID_DIR, 'grid.shp'))
    df_geo['centroid'] = df_geo['geometry'].centroid
    df_geo['centroid_lat'] = df_geo['centroid'].apply(lambda point: point.y)
    df_geo['centroid_lon'] = df_geo['centroid'].apply(lambda point: point.x)
    preds = pd.read_csv(os.path.join(CNN_GRID_OUTPUTS, f'pred_{METRIC}.csv'))

    if min_population is None:
        df_geo['to_ignore'] = False
    else:
        to_use = df_geo['population'] >= min_population
        df_geo['to_ignore'] = True
        df_geo.loc[to_use, 'to_ignore'] = False

    df_geo = merge_on_lat_lon(df_geo, preds, keys=['centroid_lat', 'centroid_lon'], how='left')

    geometry = df_geo['geometry']

    # if prediction is under 0, set to 0
    coloring_guide = df_geo[f'pred_{METRIC}']
    coloring_guide.loc[coloring_guide < 0] = 0
    vmin = max(coloring_guide.mean() - 3 * coloring_guide.std(), coloring_guide.min())
    if vmin < 0 or vmin - 0 < 0.05:
        vmin = 0

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
    vmax = None
    if METRIC == 'house_has_cellphone':
        vmax = min(1.0, coloring_guide.max())
    else:
        vmax = min(coloring_guide.mean() + 3 * coloring_guide.std(), coloring_guide.max())

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

    label = None
    if METRIC == 'house_has_cellphone':
        label = 'Predicted Device Penetration'
    elif METRIC == 'est_monthly_phone_cost_pc':
        label = 'Predicted Spend on Phone Services Per Capita ($/mo)'
    else:
        label = 'UNKNOWN'
        
    population_label = '' if min_population is None else f'\n(min. pop. {min_population})'
    ax.set_title(f'{COUNTRY_ABBRV} {label}{population_label}', fontsize=10)
    ctx.add_basemap(ax, crs=df_geo.crs)

    savepath = os.path.join(FIGURES_DIR, f'{METRIC}.png')
    print(f'Saving figure to {savepath}')
    plt.savefig(savepath)

    print('Plotting completed')


if __name__ == '__main__':
    create_folders()
    create_plot()
