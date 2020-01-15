"""
Preprocessing scripts.

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

CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__), 'script_config.ini'))
BASE_PATH = CONFIG['file_locations']['base_path']

# DATA_RAW = os.path.join(BASE_PATH, 'raw')
# DATA_PROCESSED = os.path.join(BASE_PATH, 'processed')

CONFIG = configparser.ConfigParser()
CONFIG.read('script_config.ini')

COUNTRY = CONFIG['DEFAULT']['COUNTRY']
SHAPEFILE_DIR = f'data/{COUNTRY}/shapefile'
GRID_DIR = f'data/{COUNTRY}/grid'
IMAGE_DIR = f'data/{COUNTRY}/images'


def process_country_shapes(country):
    """
    Created a set of global country shapes. Adds the single national boundary for
    each country to each country folder.

    """
    path_processed = os.path.join(SHAPEFILE_DIR, 'national_outline_{}.shp'.format(country))

    if not os.path.exists(path_processed):

        print('Working on national outline')
        path_raw = os.path.join(BASE_PATH, 'raw', 'gadm36_levels_shp', 'gadm36_0.shp')
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
        single_country = geopandas.read_file(path_processed)

    return single_country


def process_regions(country, gadm_level):
    """
    Function for processing subnational regions.

    """
    filename = 'regions_{}_{}.shp'.format(gadm_level, country)
    path_processed = os.path.join(SHAPEFILE_DIR, filename)

    if not os.path.exists(path_processed):

        print('Working on regions')
        filename = 'gadm36_{}.shp'.format(gadm_level)
        path_regions = os.path.join(BASE_PATH, 'raw', 'gadm36_levels_shp', filename)
        regions = geopandas.read_file(path_regions)

        path_countries = os.path.join(SHAPEFILE_DIR, 'national_outline_{}.shp'.format(country))
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


def process_settlement_layer(single_country, country):
    """
    """
    path_settlements = os.path.join(BASE_PATH, 'raw', 'world_pop','ppp_2020_1km_Aggregated.tif')

    settlements = rasterio.open(path_settlements)

    geo = geopandas.GeoDataFrame()

    geo = geopandas.GeoDataFrame({'geometry': single_country['geometry']}, index=[0], crs=from_epsg('4326'))

    coords = [json.loads(geo.to_json())['features'][0]['geometry']]

    #chop on coords
    out_img, out_transform = mask(settlements, coords, crop=True)

    # Copy the metadata
    out_meta = settlements.meta.copy()

    out_meta.update({"driver": "GTiff",
                    "height": out_img.shape[1],
                    "width": out_img.shape[2],
                    "transform": out_transform,
                    "crs": 'epsg:4326'})

    shape_path = os.path.join(SHAPEFILE_DIR, '{}.tif'.format(country))
    with rasterio.open(shape_path, "w", **out_meta) as dest:
            dest.write(out_img)

    return print('Completed processing of settlement layer')


def process_wb_survey_data(path):
    """
    This function takes the World Bank Living Standards Measurement
    Survey and processes all the data.

    We've used the 2016-2017 Household LSMS survey data for Malawi from
    https://microdata.worldbank.org/index.php/catalog/lsms.
    It should be in ../data/raw/LSMS/malawi-2016

    IHS4 Consumption Aggregate.csv contains:

    - Case ID: Unique household ID
    - rexpagg: Total annual per capita consumption,
        spatially & (within IHS4) temporally adjust (rexpagg)
    - adulteq: Adult equivalence
    - hh_wgt: Household sampling weight

    HouseholdGeovariablesIHS4.csv contains:

    - Case ID: Unique household ID
    - HHID: Survey solutions unique HH identifier
    - lat_modified: GPS Latitude Modified
    - lon_modified: GPS Longitude Modified

    Parameters
    ----------
    path : string
        Path to the desired data location.

    """
    ## Path to non-spatial consumption results
    file_path = os.path.join(path, 'IHS4 Consumption Aggregate.csv')

    ##Read results
    df = pd.read_csv(file_path)

    ##Estimate monthly consumption accounting for adult equivalence
    df['cons'] = df['rexpagg'] / (12 * df['adulteq'])
    df['cons'] = df['cons'] * 107.62 / (116.28 * 166.12)

    ## Rename column
    df.rename(columns={'hh_wgt': 'weight'}, inplace=True)

    ## Subset desired columns
    df = df[['case_id', 'cons', 'weight', 'urban']]

    ##Read geolocated survey data
    df_geo = pd.read_csv(os.path.join(path,
        'HouseholdGeovariables_csv/HouseholdGeovariablesIHS4.csv'))

    ##Subset household coordinates
    df_cords = df_geo[['case_id', 'HHID', 'lat_modified', 'lon_modified']]
    df_cords.rename(columns={
        'lat_modified': 'lat', 'lon_modified': 'lon'}, inplace=True)

    ##Merge to add coordinates to aggregate consumption data
    df = pd.merge(df, df_cords[['case_id', 'HHID']], on='case_id')

    ##Repeat to get df_combined
    df_combined = pd.merge(df, df_cords, on=['case_id', 'HHID'])

    ##Drop case id variable
    df_combined.drop('case_id', axis=1, inplace=True)

    ##Drop incomplete
    df_combined.dropna(inplace=True) # can't use na values

    print('Combined shape is {}'.format(df_combined.shape))

    ##Find cluster constant average
    clust_cons_avg = df_combined.groupby(
                        ['lat', 'lon']).mean().reset_index()[
                        ['lat', 'lon', 'cons']]

    ##Merge dataframes
    df_combined = pd.merge(df_combined.drop(
                        'cons', axis=1), clust_cons_avg, on=[
                        'lat', 'lon'])

    ##Get uniques
    df_uniques = df_combined.drop_duplicates(subset=
                        ['lat', 'lon'])

    print('Processed WB Living Standards Measurement Survey')

    return df_uniques, df_combined


if __name__ == '__main__':

    country = 'MWI'
    gadm_level = 3

    print('Processing national shapes')
    single_country = process_country_shapes(country)

    print('Processing subnational shapes')
    process_regions(country, gadm_level)

    print('Process settlement layer')
    process_settlement_layer(single_country, country)

    # print('Processing World Bank Living Standards Measurement Survey')
    # path = os.path.join(BASE_PATH, '..', 'lsms', 'malawi_2016')
    # df_uniques, df_combined = process_wb_survey_data(path)

    # print('Writing data')
    # df_uniques.to_csv(os.path.join(DATA_PROCESSED,
    #     'df_uniques.csv'), index=False)
    # df_combined.to_csv(os.path.join(DATA_PROCESSED,
    #     'df_combined.csv'), index=False)
