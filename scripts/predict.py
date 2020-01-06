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
import rasterio
from rasterstats import zonal_stats
from shapely.geometry import shape, mapping, Point
from collections import OrderedDict
import pyproj
from shapely.ops import transform
from tqdm import tqdm
import csv

CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__), 'script_config.ini'))
BASE_PATH = CONFIG['file_locations']['base_path']

DATA_RAW = os.path.join(BASE_PATH, 'raw')
DATA_PROCESSED = os.path.join(BASE_PATH, 'processed')


def query_nightlight_data(filepath_nl, filepath_pop, df_uniques, df_combined, path):
    """
    Query the nighlight data and export results.

    Parameters
    ----------
    filepath_nl : string
        Name of the nightlight file to load.
    df_uniques : dataframe
        All unique survey locations.
    df_combined : dataframe
        All household survey locations.
    path : string
        Path to the desired data location.

    """
    results = []
    shapes = []

    for i, r in df_uniques.iterrows():

        #create shapely object
        lat = r.lat#.values[0]
        lon = r.lon#.values[0]
        geom = Point(lon, lat)

        #transform to projected coordinate
        project = pyproj.Transformer.from_proj(
            pyproj.Proj('epsg:4326'), # source coordinate system
            pyproj.Proj('epsg:3857')) # destination coordinate system
        geom_transformed = transform(project.transform, geom)

        #add 10km2 buffer
        buffer_1km = geom_transformed.buffer(500, cap_style=3)
        buffer_10km = geom_transformed.buffer(1580, cap_style=3)

        #get area
        area_1km2 = round(buffer_1km.area / 1e6, 1)
        area_10km2 = round(buffer_10km.area / 1e6, 1)

        #transform to unprojected
        project = pyproj.Transformer.from_proj(
            pyproj.Proj('epsg:3857'), # source coordinate system
            pyproj.Proj('epsg:4326')) # destination coordinate system
        geom_1km = transform(project.transform, buffer_1km)
        geom_10km = transform(project.transform, buffer_10km)

        #get population
        luminosity_sum_1km = zonal_stats(geom_1km, filepath_nl, stats="sum", nodata=0)[0]['sum']
        population_1km = zonal_stats(geom_1km, filepath_pop, stats="sum", nodata=0)[0]['sum']

        #get population
        luminosity_sum_10km = zonal_stats(geom_10km, filepath_nl, stats="sum", nodata=0)[0]['sum']
        population_10km = zonal_stats(geom_10km, filepath_pop, stats="sum", nodata=0)[0]['sum']

        results.append({
            'HHID': r.HHID,
            'lat': lat,
            'lon': lon,
            'luminosity_sum_1km': catch_missing_values(luminosity_sum_1km),
            'population_1km': catch_missing_values(population_1km),
            'area_km2_1km': area_1km2,
            'luminosity_sum_10km': catch_missing_values(luminosity_sum_10km),
            'population_10km': catch_missing_values(population_10km),
            'area_km2_10km': area_10km2,
        })

        shapes.append({
            'type': 'Polygon',
            'geometry': mapping(geom_1km),
            'properties': {}
        })

        shapes.append({
            'type': 'Polygon',
            'geometry': mapping(geom_10km),
            'properties': {}
        })

    results = pd.DataFrame(results)

    df_uniques = pd.merge(df_uniques, results, on=['lat', 'lon', 'HHID'])

    df_combined = pd.merge(df_combined, results, on=['lat', 'lon', 'HHID'])

    return df_combined, shapes


def catch_missing_values(value):
    """
    Catch missing / none / NaN values and replace with 0.

    """
    try:
        if value > 0:
            return value
        else:
            return 0
    except:
        return 0


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
        lambda x: 'rural' if x == 0 else 'urban')

    clust_averages = clust_averages.drop('urban', axis=1)

    clust_averages.rename(columns={'urban_encoded': 'urban'}, inplace=True)

    return clust_averages


def get_coefficients(df):
    """
    Calculate correlation coefficient using np.corrcoef.

    Parameters
    ----------
    x : array
        Array of numeric values.
    y : array
        Array of numeric values.

    """
    output = {}

    urban = df.loc[df['urban'] == 'urban']
    x = urban.cons
    y = urban.luminosity_sum_1km
    output['urban_1km'] = np.corrcoef(x, y)[0, 1]**2

    rural = df.loc[df['urban'] == 'rural']
    x = rural.cons
    y = rural.luminosity_sum_1km
    output['rural_1km'] = np.corrcoef(x, y)[0, 1]**2

    urban = df.loc[df['urban'] == 'urban']
    x = urban.cons
    y = urban.luminosity_sum_10km
    output['urban_10km'] = np.corrcoef(x, y)[0, 1]**2

    rural = df.loc[df['urban'] == 'rural']
    x = rural.cons
    y = rural.luminosity_sum_10km
    output['rural_10km'] = np.corrcoef(x, y)[0, 1]**2

    return output


def load_grid(path):
    """
    Load in grid produced in scripts/grid.py

    """
    grid = []

    with fiona.open(path, 'r') as source:
        for item in source:
            grid.append(item)

    return grid


def get_geom_area_km2(geom, old_crs, new_crs):
    """
    Transform to epsg: 3859.

    """
    project = pyproj.Transformer.from_proj(
        pyproj.Proj(old_crs), # source coordinate system
        pyproj.Proj(new_crs)) # destination coordinate system

    geom_transformed = transform(project.transform, geom)

    area_km2 = (geom_transformed.area / 1e6)

    return area_km2


def query_data(grid, filepath_nl, filepath_lc, filepath_pop, coefficients):
    """
    Query raster layer for each shape in grid.

    """
    output = []
    csv = []

    for grid_area in tqdm(grid):

        geom = shape(grid_area['geometry'])

        landcover = zonal_stats(geom.centroid, filepath_lc, stats="sum", nodata=0)[0]['sum']

        if landcover == 13:
            urban = 1
        else:
            urban = 0

        population = zonal_stats(geom.centroid, filepath_pop, stats="sum", nodata=0)[0]['sum']

        area_km2 = get_geom_area_km2(geom, 'epsg:4326', 'epsg:3857')

        try:
            if population > 0:
                pop_density_km2 = population / area_km2
            else:
                pop_density_km2 = 0
        except:
            pop_density_km2 = 0

        if pop_density_km2 >= 2500:
            urban = 1
        else:
            urban = 0

        stats = zonal_stats(geom, filepath_nl, stats="mean sum", nodata=0)

        try:
            if float(stats[0]['sum']) > 0:
                if urban == 1:
                    pred_consumption = float(stats[0]['sum']) * (1 + float(coefficients['urban']))
                else:
                    pred_consumption = float(stats[0]['sum']) * (1 + float(coefficients['rural']))
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
                'population': population,
                'pop_density_km2': pop_density_km2,
                'area_km2': area_km2,
                'luminosity_mean': luminosity_mean,
                'luminosity_sum': luminosity_sum,
                'pred_consumption': pred_consumption,
                'landcover': landcover,
                'urban': urban,
            }
        })

        csv.append({
            'population': population,
            'pop_density_km2': pop_density_km2,
            'area_km2': area_km2,
            'luminosity_mean': luminosity_mean,
            'luminosity_sum': luminosity_sum,
            'pred_consumption': pred_consumption,
            'landcover': landcover,
            'urban': urban,
        })

    return output, csv


def csv_writer(data, directory, filename):
    """
    Write data to a CSV file path.
    Parameters
    ----------
    data : list of dicts
        Data to be written.
    directory : string
        Path to export folder
    filename : string
        Desired filename.
    """
    # Create path
    if not os.path.exists(directory):
        os.makedirs(directory)

    fieldnames = []
    for name, value in data[0].items():
        fieldnames.append(name)

    with open(os.path.join(directory, filename), 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames, lineterminator = '\n')
        writer.writeheader()
        writer.writerows(data)


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

    print('Defining nightlight data path')
    year = 2013
    folder_name = 'noaa_dmsp_ols_nightlight_data'
    path_nl = os.path.join(DATA_RAW, folder_name)
    filename = 'F182013.v4c_web.stable_lights.avg_vis.tif'
    filepath_nl = os.path.join(path_nl, str(year), filename)

    print('Defining world pop data path')
    folder_name = 'world_pop'
    path_pop = os.path.join(DATA_RAW, folder_name)
    filename = 'ppp_2020_1km_Aggregated.tif'
    filepath_pop = os.path.join(path_pop, filename)

    print('Read df_uniques.csv')
    df_uniques = pd.read_csv(os.path.join(DATA_PROCESSED, 'df_uniques.csv'))

    print('Read df_combined.csv')
    df_combined = pd.read_csv(os.path.join(DATA_PROCESSED, 'df_combined.csv'))

    print('Querying nightlight data')
    df_combined, shapes = query_nightlight_data(filepath_nl, filepath_pop,
        df_uniques, df_combined, os.path.join(path_nl, str(year)))

    print('Creating clusters')
    clust_averages = create_clusters(df_combined)

    print('Remove extreme values')
    clust_averages = clust_averages.drop(clust_averages[clust_averages.cons > 2000].index)

    print('Getting coefficient value')
    coefficients = get_coefficients(clust_averages)

    print('Predict consumption using nightlights')
    urban = clust_averages.loc[clust_averages['urban'] == 'urban']
    urban['cons_pred_1km'] = urban['luminosity_sum_1km'] * (1 + coefficients['urban_1km'])

    rural = clust_averages.loc[clust_averages['urban'] == 'rural']
    rural['cons_pred_1km'] = rural['luminosity_sum_1km'] * (1 + coefficients['rural_1km'])

    clust_averages_1km = [urban, rural]
    clust_averages_1km = pd.concat(clust_averages_1km)

    urban = clust_averages.loc[clust_averages['urban'] == 'urban']
    urban['cons_pred_10km'] = urban['luminosity_sum_10km'] * (1 + coefficients['urban_10km'])

    rural = clust_averages.loc[clust_averages['urban'] == 'rural']
    rural['cons_pred_10km'] = rural['luminosity_sum_10km'] * (1 + coefficients['rural_10km'])

    clust_averages_10km = [urban, rural]
    clust_averages_10km = pd.concat(clust_averages_10km)

    clust_averages = [clust_averages_1km, clust_averages_10km]
    clust_averages = pd.concat(clust_averages)

    print('Writing all other data')
    clust_averages.to_csv(os.path.join(DATA_PROCESSED,
        'lsms-cluster-2016.csv'), index=False)

    print('Loading grid')
    path = os.path.join(DATA_PROCESSED, 'grid_test.shp')
    grid = load_grid(path)

    print('Defining land cover data path')
    folder_name = 'modis_landcover'
    path_lc = os.path.join(DATA_RAW, folder_name)
    filename = 'MCD12Q1.006_LC_Type1_doy2016001_aid0001.tif'
    filepath_lc = os.path.join(path_lc,filename)

    print('Querying nightlights using grid')
    grid, csv_data = query_data(grid, filepath_nl, filepath_lc, filepath_pop,
        coefficients)

    print('Writing outputs to results folder')
    path_results = os.path.join(BASE_PATH, '..', 'results')
    csv_writer(csv_data, path_results, 'results.csv')
    write_shapefile(grid, path_results, 'results.shp', 'epsg:4326')

    write_shapefile(shapes, path_results, 'buffer_boxes.shp', 'epsg:4326')

    coeffs_to_write = []
    coeffs_to_write.append(coefficients)
    csv_writer(coeffs_to_write, path_results, 'coefficients.csv')
