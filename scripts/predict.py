"""
Preprocessing scripts.

Written by Jatin Mathur and Ed Oughton.

Winter 2020

"""
import os
import configparser
import pandas as pd
import numpy as np
import glob
import geoio
import math
import geopandas as gpd
import fiona
from rasterstats import zonal_stats
from shapely.geometry import shape, mapping
from collections import OrderedDict

CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__), 'script_config.ini'))
BASE_PATH = CONFIG['file_locations']['base_path']

DATA_RAW = os.path.join(BASE_PATH, 'raw')
DATA_PROCESSED = os.path.join(BASE_PATH, 'processed')


def query_nightlight_data(filename, df_uniques, df_combined, path):
    """
    Query the nighlight data and export results.

    Parameters
    ----------
    filename : string
        Name of the nightlight file to load.
    df_uniques : dataframe
        All unique survey locations.
    df_combined : dataframe
        All household survey locations.
    path : string
        Path to the desired data location.

    """
    img = geoio.GeoImage(filename)
    ##Convert points in projection space to points in raster space.
    xPixel, yPixel = img.proj_to_raster(34.915074, -14.683761)

    ##Remove single-dimensional entries from the shape of an array.
    im_array = np.squeeze(img.get_data())

    ##Get the nightlight values
    im_array[int(yPixel),int(xPixel)]

    household_nightlights = []
    for i,r in df_uniques.iterrows():

        ##Create 10km^2 bounding box around point
        min_lat, min_lon, max_lat, max_lon = create_space(r.lat, r.lon)

        ##Convert point coordinaces to raster space
        xminPixel, yminPixel = img.proj_to_raster(min_lon, min_lat)
        xmaxPixel, ymaxPixel = img.proj_to_raster(max_lon, max_lat)

        ##Get min max values
        xminPixel, xmaxPixel = (min(xminPixel, xmaxPixel),
                                max(xminPixel, xmaxPixel))
        yminPixel, ymaxPixel = (min(yminPixel, ymaxPixel),
                                max(yminPixel, ymaxPixel))
        xminPixel, yminPixel, xmaxPixel, ymaxPixel = (
                                int(xminPixel), int(yminPixel),
                                int(xmaxPixel), int(ymaxPixel))

        ##Append mean value data to df
        household_nightlights.append(
            im_array[yminPixel:ymaxPixel,xminPixel:xmaxPixel].mean())

    df_uniques['nightlights'] = household_nightlights

    df_combined = pd.merge(df_combined, df_uniques[
                    ['lat', 'lon', 'nightlights']], on=['lat', 'lon'])

    print('Complete querying process')

    return df_combined


def create_space(lat, lon):
    """
    Creates a 10km^2 area bounding box.

    Parameters
    ----------
    lat : float
        Latitude
    lon : float
        Longitude

    """
    bottom = lat - (180 / math.pi) * (5000 / 6378137)
    left = lon - (180 / math.pi) * (5000 / 6378137) / math.cos(lat)
    top = lat + (180 / math.pi) * (5000 / 6378137)
    right = lon + (180 / math.pi) * (5000 / 6378137) / math.cos(lat)

    return bottom, left, top, right


def create_clusters(df_combined):
    """
    Create cluster locations and allocate settlement type.

    Parameters
    ----------
    df_combined : dataframe

    """
    # encode "RURAL" as 0 and "URBAN" as 1
    df_combined['urban_encoded'] = pd.factorize(df_combined['urban'])[0]

    clust_groups = df_combined.groupby(['lat', 'lon'])

    clust_averages = clust_groups.mean().reset_index()

    counts = clust_groups.count().reset_index()[['lat', 'lon', 'cons']]
    counts.rename(columns={'cons': 'num_households'}, inplace=True)
    clust_averages = pd.merge(clust_averages, counts, on=['lat', 'lon'])

    # if more than 0.5 average within a clust, label it as 1 (URBAN), else 0
    clust_averages['urban_encoded'] = clust_averages['urban_encoded'].apply(
        lambda x: round(x))

    clust_averages['urban_encoded'] = clust_averages['urban_encoded'].apply(
        lambda x: 'Rural' if x == 0 else 'Urban')

    clust_averages = clust_averages.drop('urban', axis=1)

    clust_averages.rename(columns={'urban_encoded': 'urban'}, inplace=True)

    return clust_averages


def get_r2_numpy_corrcoef(df, subset_value):
    """
    Calculate correlation coefficient using np.corrcoef.

    Parameters
    ----------
    x : array
        Array of numeric values.
    y : array
        Array of numeric values.

    """
    subset = df.loc[df['urban'] == subset_value]
    x = subset.cons
    y = subset.nightlights

    return np.corrcoef(x, y)[0, 1]**2


def load_grid(path):
    """
    Load in grid produced in scripts/grid.py

    """
    grid = []

    with fiona.open(path, 'r') as source:
        for item in source:
            grid.append(item)

    return grid


def query_data(grid, filepath_nl, filepath_lc, nl_coefficient_urban, nl_coefficient_rural):
    """
    Query raster layer for each shape in grid.

    """
    output = []

    for grid_area in grid:

        #query
        geom = shape(grid_area['geometry'])

        landcover = zonal_stats(geom.centroid, filepath_lc, stats="sum")[0]['sum']

        if landcover == 13:
            urban = 1
        else:
            urban = 0

        stats = zonal_stats(geom, filepath_nl, stats="mean sum")

        try:
            if float(stats[0]['sum']) > 0:
                if urban == 1:
                    pred_consumption = float(stats[0]['sum']) * (1 + float(nl_coefficient_urban))
                else:
                    pred_consumption = float(stats[0]['sum']) * (1 + float(nl_coefficient_rural))
            else:
                pred_consumption = 0
        except:
            pred_consumption = 0

        try:
            if float(stats[0]['mean']) > 0:
                luminosity_mean = stats[0]['mean']
            else:
                luminosity_mean = 0
        except:
            luminosity_mean = 0

        try:
            if float(stats[0]['sum']) > 0:
                luminosity_sum = stats[0]['sum']
            else:
                luminosity_sum = 0
        except:
            luminosity_sum = 0


        output.append({
            'type': grid_area['type'],
            'geometry': mapping(geom),
            'id': grid_area['id'],
            'properties': {
                'luminosity_mean': luminosity_mean,
                'luminosity_sum': luminosity_sum,
                'pred_consumption': pred_consumption,
                'landcover': landcover,
                'urban': urban,
            }
        })

    return output


def write_shapefile(data, directory, filename, crs):
    """
    Write geojson data to shapefile.
    """
    prop_schema = []
    for name, value in data[0]['properties'].items():
        fiona_prop_type = next((
            fiona_type for fiona_type, python_type in \
                fiona.FIELD_TYPES_MAP.items() if \
                python_type == type(value)), None
            )

        prop_schema.append((name, fiona_prop_type))

    sink_driver = 'ESRI Shapefile'
    sink_crs = {'init': crs}
    sink_schema = {
        'geometry': data[0]['geometry']['type'],
        'properties': OrderedDict(prop_schema)
    }

    if not os.path.exists(directory):
        os.makedirs(directory)

    with fiona.open(
        os.path.join(directory, filename), 'w',
        driver=sink_driver, crs=sink_crs, schema=sink_schema) as sink:
        for datum in data:
            sink.write(datum)


if __name__ == '__main__':

    year = 2013
    folder_name = 'noaa_dmsp_ols_nightlight_data'
    path_nl = os.path.join(DATA_RAW, folder_name)
    filename = 'F182013.v4c_web.stable_lights.avg_vis.tif'
    filepath_nl = os.path.join(path_nl, str(year), filename)

    print('Read df_uniques.csv')
    df_uniques = pd.read_csv(os.path.join(DATA_PROCESSED, 'df_uniques.csv'))

    print('Read df_combined.csv')
    df_combined = pd.read_csv(os.path.join(DATA_PROCESSED, 'df_combined.csv'))

    print('Querying nightlight data')
    df_combined = query_nightlight_data(filepath_nl, df_uniques,
        df_combined, os.path.join(path_nl, str(year)))

    print('Creating clusters')
    clust_averages = create_clusters(df_combined)

    print('Remove extreme values')
    clust_averages = clust_averages.drop(clust_averages[clust_averages.cons > 50].index)

    print('Getting coefficient value')
    nl_coefficient_urban = get_r2_numpy_corrcoef(clust_averages, 'Urban')
    nl_coefficient_rural = get_r2_numpy_corrcoef(clust_averages, 'Rural')

    print('Predict consumption using nightlights')
    urban = clust_averages.loc[clust_averages['urban'] == 'Urban']
    urban['cons_pred'] = urban['nightlights'] * (1 + nl_coefficient_urban)

    rural = clust_averages.loc[clust_averages['urban'] == 'Rural']
    rural['cons_pred'] = rural['nightlights'] * (1 + nl_coefficient_rural)

    clust_averages = [urban, rural]
    clust_averages = pd.concat(clust_averages)

    print('Writing all other data')
    clust_averages.to_csv(os.path.join(DATA_PROCESSED,
        'lsms-cluster-2016.csv'), index=False)

    print('Loading grid')
    path = os.path.join(DATA_PROCESSED, 'grid_test.shp')
    grid = load_grid(path)

    print('Querying nightlights using grid')

    folder_name = 'modis_landcover'
    path_lc = os.path.join(DATA_RAW, folder_name)
    filename = 'MCD12Q1.006_LC_Type1_doy2016001_aid0001.tif'
    filepath_lc = os.path.join(path_lc,filename)

    grid = query_data(grid, filepath_nl, filepath_lc, nl_coefficient_urban, nl_coefficient_rural)

    print('Writing outputs to results folder')
    path_results = os.path.join(BASE_PATH, '..', 'results')
    write_shapefile(grid, path_results, 'results.shp', 'EPSG:4326')
