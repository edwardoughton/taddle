"""
Preprocessing scripts.

Written by Jatin Mathur and Ed Oughton.

Winter 2020

"""
import os
import configparser
import requests
import tarfile
import gzip
import shutil

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
