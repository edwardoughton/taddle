"""
Preprocessing scripts.

Written by Ed Oughton.

Winter 2020

"""
import os
import configparser
import pandas as pd
import geopandas

from shapely.geometry import MultiPolygon, Polygon, mapping, box

CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__), 'script_config.ini'))
BASE_PATH = CONFIG['file_locations']['base_path']

DATA_RAW = os.path.join(BASE_PATH, 'raw')
DATA_PROCESSED = os.path.join(BASE_PATH, 'processed')

def process_country_shapes(country):
    """
    Created a set of global country shapes. Adds the single national boundary for
    each country to each country folder.

    """
    path_processed = os.path.join(DATA_PROCESSED, 'national_outline_{}.shp'.format(country))

    if not os.path.exists(path_processed):

        print('Working on national outline')
        path_raw = os.path.join(DATA_RAW, 'gadm36_levels_shp', 'gadm36_0.shp')
        countries = geopandas.read_file(path_raw)

        for name in countries.GID_0.unique():

            if not name == country:
                continue

            print('Working on {}'.format(name))
            single_country = countries[countries.GID_0 == name]

            print('Excluding small shapes')
            single_country['geometry'] = single_country.apply(exclude_small_shapes,axis=1)

            print('Simplifying geometries')
            single_country['geometry'] = single_country.simplify(tolerance = 0.005,
                preserve_topology=True).buffer(0.01).simplify(tolerance = 0.005,
                preserve_topology=True)

            print('Writing national outline to file')
            single_country.to_file(path_processed, driver='ESRI Shapefile')

    else:
        countries = geopandas.read_file(path_processed)

    return print('Completed processing of country shapes')


def process_regions(country, gadm_level):
    """
    Function for processing subnational regions.

    """
    filename = 'regions_{}_{}.shp'.format(gadm_level, country)
    path_processed = os.path.join(DATA_PROCESSED, filename)

    if not os.path.exists(path_processed):

        print('Working on regions')
        filename = 'gadm36_{}.shp'.format(gadm_level)
        path_regions = os.path.join(DATA_RAW, 'gadm36_levels_shp', filename)
        regions = geopandas.read_file(path_regions)

        path_countries = os.path.join(DATA_PROCESSED, 'national_outline_{}.shp'.format(country))
        countries = geopandas.read_file(path_countries)

        for name in countries.GID_0.unique():

            if not name == country:
                continue

            print('Working on {}'.format(name))
            regions = regions[regions.GID_0 == name]

            print('Excluding small shapes')
            regions['geometry'] = regions.apply(exclude_small_shapes,axis=1)

            print('Simplifying geometries')
            regions['geometry'] = regions.simplify(tolerance = 0.005, preserve_topology=True) \
                .buffer(0.01).simplify(tolerance = 0.005, preserve_topology=True)

            print('Writing global_regions.shp to file')
            regions.to_file(path_processed, driver='ESRI Shapefile')

        print('Completed processing of regional shapes level {}'.format(gadm_level))

    else:
        regions = geopandas.read_file(path_processed)

    return regions


def exclude_small_shapes(x,regionalized=False):
    """
    This function will remove the small shapes of multipolygons. Will reduce the size
        of the file.

    Arguments:
        *x* : a geometry feature (Polygon) to simplify. Countries which are very large will
        see larger (unhabitated) islands being removed.

    Optional Arguments:
        *regionalized*  : Default is **False**. Set to **True** will use lower threshold
        settings (default: **False**).

    Returns:
        *MultiPolygon* : a shapely geometry MultiPolygon without tiny shapes.

    """
    # if its a single polygon, just return the polygon geometry
    if x.geometry.geom_type == 'Polygon':
        return x.geometry

    # if its a multipolygon, we start trying to simplify and remove shapes if its too big.
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

        # save remaining polygons as new multipolygon for the specific country
        new_geom = []
        for y in x.geometry:
            if y.area > threshold:
                new_geom.append(y)

        return MultiPolygon(new_geom)


if __name__ == '__main__':

    country = 'MWI'
    gadm_level = 3

    process_country_shapes(country)

    process_regions(country, gadm_level)
