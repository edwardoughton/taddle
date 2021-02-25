"""
Extract and preprocess shapefile for chosen country using GADM data

Written by Ed Oughton.

Winter 2020
"""

import os
import configparser
import pandas as pd
import geopandas
import rasterio
from rasterio.mask import mask
import json
from fiona.crs import from_epsg
from shapely.geometry import MultiPolygon, Polygon, mapping, box

BASE_DIR = '.'
# repo imports
import sys
sys.path.append(BASE_DIR)
from config import VIS_CONFIG

COUNTRY_ABBRV = VIS_CONFIG['COUNTRY_ABBRV']
COUNTRIES_DIR = os.path.join(BASE_DIR, 'data', 'countries')
SHAPEFILE_DIR = os.path.join(COUNTRIES_DIR, COUNTRY_ABBRV, 'shapefile')


def create_folders():
    os.makedirs(SHAPEFILE_DIR, exist_ok=True)


def process_country_shapes():
    """
    Created a set of global country shapes. Adds the single
    national boundary for each country to each country folder.
    """
    path_processed = os.path.join(
        SHAPEFILE_DIR, 'national_outline_{}.shp'.format(COUNTRY_ABBRV))

    single_country = None
    if not os.path.exists(path_processed):
        print('Working on national outline')
        path_raw = os.path.join(BASE_DIR, 'data', 'gadm36_levels_shp', 'gadm36_0.shp')
        countries = geopandas.read_file(path_raw)

        for name in countries.GID_0.unique():
            if not name == COUNTRY_ABBRV:
                continue

            print('Working on {}'.format(name))
            single_country = countries[countries.GID_0 == name]

            print('Excluding small shapes')
            single_country['geometry'] = single_country.apply(
                exclude_small_shapes,axis=1)

            print('Simplifying geometries')
            single_country['geometry'] = single_country.simplify(
                tolerance = 0.005, preserve_topology=True
                ).buffer(0.01).simplify(tolerance = 0.005,
                    preserve_topology=True)

            print('Writing national outline to file')
            single_country.to_file(path_processed, driver='ESRI Shapefile')
            found = True
            break
        
        if not found:
            raise ValueError(f'country abbrv {COUNTRY_ABBRV} does not exist')

    else:
        single_country = geopandas.read_file(path_processed)

    return single_country


def process_regions(gadm_level):
    """
    Function for processing subnational regions.
    """
    filename = 'regions_{}_{}.shp'.format(gadm_level, COUNTRY_ABBRV)
    path_processed = os.path.join(SHAPEFILE_DIR, filename)

    if not os.path.exists(path_processed):

        print('Working on regions')
        filename = 'gadm36_{}.shp'.format(gadm_level)
        path_regions = os.path.join(
            'data', 'gadm36_levels_shp', filename)
        regions = geopandas.read_file(path_regions)

        path_countries = os.path.join(SHAPEFILE_DIR,
            'national_outline_{}.shp'.format(COUNTRY_ABBRV))
        countries = geopandas.read_file(path_countries)

        for name in countries.GID_0.unique():

            if not name == COUNTRY_ABBRV:
                continue

            print('Working on {}'.format(name))
            regions = regions[regions.GID_0 == name]

            print('Excluding small shapes')
            regions['geometry'] = regions.apply(exclude_small_shapes,axis=1)

            print('Simplifying geometries')
            regions['geometry'] = regions.simplify(
                tolerance = 0.005, preserve_topology=True) \
                .buffer(0.01).simplify(tolerance = 0.005,
                preserve_topology=True)

            print('Writing global_regions.shp to file')
            regions.to_file(path_processed, driver='ESRI Shapefile')

        print('Completed processing of regional shapes level {}'.format(gadm_level))

    else:
        regions = geopandas.read_file(path_processed)

    return regions


def exclude_small_shapes(x,regionalized=False):
    """
    This function will remove the small shapes of multipolygons.
    Will reduce the size of the file.

    Arguments:
        *x* : a geometry feature (Polygon) to simplify.
        Countries which are very large will see larger (unhabitated)
        islands being removed.

    Optional Arguments:
        *regionalized*  : Default is **False**. Set to **True** will
        use lower threshold settings (default: **False**).

    Returns:
        *MultiPolygon* : a shapely geometry MultiPolygon without
        tiny shapes.
    """
    # if its a single polygon, just return the polygon geometry
    if x.geometry.geom_type == 'Polygon':
        return x.geometry

    # if its a multipolygon, we start trying to simplify and
    # remove shapes if its too big.
    elif x.geometry.geom_type == 'MultiPolygon':

        if regionalized == False:
            area1 = 0.1
            area2 = 250

        elif regionalized == True:
            area1 = 0.01
            area2 = 50

        # dont remove shapes if total area is already very small
        if x.geometry.area < area1:
            return x.geometry
        # remove bigger shapes if country is really big

        if x['GID_0'] in ['CHL','IDN']:
            threshold = 0.01
        elif x['GID_0'] in ['RUS','GRL','CAN','USA']:
            if regionalized == True:
                threshold = 0.01
            else:
                threshold = 0.01

        elif x.geometry.area > area2:
            threshold = 0.1
        else:
            threshold = 0.001

        # save remaining polygons as new multipolygon for the
        # specific country
        new_geom = []
        for y in x.geometry:
            if y.area > threshold:
                new_geom.append(y)

        return MultiPolygon(new_geom)


def process_settlement_layer(single_country):
    """
    """
    path_settlements = os.path.join(
        BASE_DIR, 'data', 'world_population','ppp_2020_1km_Aggregated.tif')

    settlements = rasterio.open(path_settlements)

    geo = geopandas.GeoDataFrame()

    geo = geopandas.GeoDataFrame({
        'geometry': single_country['geometry']}, index=[0],
        crs=from_epsg('4326'))

    coords = [json.loads(geo.to_json())['features'][0]['geometry']]

    #chop on coords
    out_img, out_transform = mask(settlements, coords, crop=True)

    # Copy the metadata
    out_meta = settlements.meta.copy()

    out_meta.update({"driver": "GTiff",
                    "height": out_img.shape[1],
                    "width": out_img.shape[2],
                    "transform": out_transform,
                    "CRS": 'epsg:4326'})

    shape_path = os.path.join(SHAPEFILE_DIR, '{}.tif'.format(COUNTRY_ABBRV))
    with rasterio.open(shape_path, "w", **out_meta) as dest:
        dest.write(out_img)

    print('Completed processing of settlement layer')
    return


if __name__ == '__main__':
    gadm_level = 3
    create_folders()

    print('Processing national shapes')
    single_country = process_country_shapes()

    print('Processing subnational shapes')
    process_regions(gadm_level)

    print('Process settlement layer')
    process_settlement_layer(single_country)
