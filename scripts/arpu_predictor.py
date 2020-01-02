"""
Preprocessing scripts.

Written by Jatin Mathur and Ed Oughton.

Winter 2020

"""
import os
import configparser
import pandas as pd
import numpy as np
import requests
import tarfile
import gzip
import shutil
import glob
import geoio
import math

CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__), 'script_config.ini'))
BASE_PATH = CONFIG['file_locations']['base_path']

DATA_RAW = os.path.join(BASE_PATH, 'raw')
DATA_PROCESSED = os.path.join(BASE_PATH, 'processed')


def get_nightlight_data(folder_name, path, data_year):
    """
    Downloads the nighlight data from NOAA.

    As these files are large, they can take a couple of minutes to download.

    Parameters
    ----------
    path : string
        Path to the desired data location.
    data_year : int
        The desired year of the chosen dataset (default: 2013).

    """
    if not os.path.exists(path):
        os.makedirs(path)

    for year in [data_year]:

        year = str(year)
        url = ('https://ngdc.noaa.gov/eog/data/web_data/v4composites/F18'
                + year + '.v4.tar')
        target = os.path.join(path, year)

        if not os.path.exists(target):
            os.makedirs(target, exist_ok=True)

        target += '/' + folder_name
        response = requests.get(url, stream=True)

        if not os.path.exists(target):
            if response.status_code == 200:
                print('Downloading data')
                with open(target, 'wb') as f:
                    f.write(response.raw.read())

    print('Data download complete')

    for year in [data_year]:

        print('Working on {}'.format(year))
        folder_loc = os.path.join(path, str(year))
        file_loc = os.path.join(folder_loc, folder_name)

        print('Unzipping data')
        tar = tarfile.open(file_loc)
        tar.extractall(path=folder_loc)

        files = os.listdir(folder_loc)
        for filename in files:
            file_path = os.path.join(path, str(year), filename)
            if 'stable' in filename: # only need stable_lights
                if file_path.split('.')[-1] == 'gz':
                    # unzip the file is a .gz file
                    with gzip.open(file_path, 'rb') as f_in:
                        with open(file_path[:-3], 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)

    return print('Downloaded and processed night light data')


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

    ##Estimate daily consumption accounting for adult equivalence
    df['cons'] = df['rexpagg'] / (365 * df['adulteq'])
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


def get_r2_numpy_corrcoef(x, y):
    """
    Calculate correlation coefficient using np.corrcoef.

    Parameters
    ----------
    x : array
        Array of numeric values.
    y : array
        Array of numeric values.

    """
    return np.corrcoef(x, y)[0, 1]**2


if __name__ == '__main__':

    year = 2013
    folder_name = 'noaa_dmsp_ols_nightlight_data'
    path_nightlights = os.path.join(DATA_RAW, folder_name)
    filename = 'F182013.v4c_web.stable_lights.avg_vis.tif'
    filepath = os.path.join(path_nightlights, str(year), filename)

    if not os.path.exists(filepath):
        print('Need to download nightlight data first')
        get_nightlight_data(folder_name, path_nightlights, year)
    else:
        print('Nightlight data already exists in data folder')

    print('Processing World Bank Living Standards Measurement Survey')
    path = os.path.join(DATA_RAW, 'lsms', 'malawi_2016')
    df_uniques, df_combined = process_wb_survey_data(path)

    print('Querying nightlight data')
    df_combined = query_nightlight_data(filepath, df_uniques,
        df_combined, os.path.join(path_nightlights, str(year)))

    print('Creating clusters')
    clust_averages = create_clusters(df_combined)

    print('Remove extreme values')
    clust_averages = clust_averages.drop(clust_averages[clust_averages.cons > 50].index)

    print('Getting coefficient value')
    nl_coefficient = get_r2_numpy_corrcoef(clust_averages.cons, clust_averages.nightlights)

    print('Predict consumption using nightlights')
    clust_averages['cons_pred'] = clust_averages['nightlights'] * (1 + nl_coefficient)

    print('Writing all other data')
    clust_averages.to_csv(os.path.join(DATA_PROCESSED,
        'lsms-cluster-2016.csv'), index=False)
