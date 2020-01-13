import requests
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
import pandas as pd

def merge_on_lat_lon(df1, df2, keys=['cluster_lat', 'cluster_lon']):
    """
        Allows two dataframes to be merged on lat/lon
        Necessary because pandas has trouble merging on floats
    """
    df1 = df1.copy()
    df2 = df2.copy()
    
    # must use ints for merging, as floats induce errors
    df1['merge_lat'] = (10000 * df1[keys[0]]).astype(int)
    df1['merge_lon'] = (10000 * df1[keys[1]]).astype(int)
    
    df2['merge_lat'] = (10000 * df2[keys[0]]).astype(int)
    df2['merge_lon'] = (10000 * df2[keys[1]]).astype(int)
    
    df2.drop(keys, axis=1, inplace=True)
    merged = pd.merge(df1, df2, on=['merge_lat', 'merge_lon'])
    merged.drop(['merge_lat', 'merge_lon'], axis=1, inplace=True)
    return merged

class RidgeEnsemble:
    def __init__(self, ridges, scalers):
        assert type(ridges) == list and type(scalers) == list and len(ridges) == len(scalers)
        self.ridges = ridges
        self.scalers = scalers
    
    def predict(self, x):
        predictions = np.zeros(len(x))
        for ridge, scalar in zip(self.ridges, self.scalers):
            feats = scalar.transform(x)
            predictions += ridge.predict(feats)
        predictions /= len(self.ridges)
        return predictions

class ImageryDownloader:
    def __init__(self, access_token):
        self.access_token = access_token
        self.url = 'https://maps.googleapis.com/maps/api/staticmap?center={},{}&zoom={}&size=400x400&maptype=satellite&key={}'
    
    def download(self, lat, long, zoom):
        res = requests.get(self.url.format(lat, long, zoom, self.access_token))
        # server needs to make image available, takes a few seconds
        if res.status_code == 403:
            return 'RETRY'
        assert res.status_code < 400, print(f'Error - failed to download {lat}, {long}, {zoom}')
        image = plt.imread(BytesIO(res.content))
        return image
    
class CustomProgressBar:
    """
    Made a custom progress "bar" because tqdm was slowing down my forward pass speed by a 2x factor...
    Really just a rudimentary carriage return printer :)
    """
    def __init__(self, total):
        self.total = total
        self.cur = 0
        
    def update(self, amount):
        self.cur += amount
        print(f'\t {round(self.cur*100/self.total, 2)}%', end='\r')

