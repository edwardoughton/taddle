"""
Pass downloaded images through the model pipeline, namely:
    - extract features using CNN
    - aggregate cluster-level features
    - use ridge models to predict phone_consumption, phone_density, or consumption

Written by Jatin Mathur

Winter 2020

"""

import configparser
import torch
import torch.nn as nn
from torchvision import transforms
import pandas as pd
import numpy as np
import os
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib
import logging
import warnings
warnings.filterwarnings('ignore')

# repo imports
import sys
sys.path.append('.')
from utils import RidgeEnsemble, CustomProgressBar

CONFIG = configparser.ConfigParser()
CONFIG.read('script_config.ini')

COUNTRY = CONFIG['DEFAULT']['COUNTRY']
GRID_DIR = f'data/{COUNTRY}/grid'
IMAGE_DIR = f'data/{COUNTRY}/images'

CNN_DIR = CONFIG['MODELS']['CNN_DIR']
RIDGE_PHONE_DENSITY_DIR = CONFIG['MODELS']['RIDGE_PHONE_DENSITY_DIR']
RIDGE_PHONE_CONSUMPTION_DIR = CONFIG['MODELS']['RIDGE_PHONE_CONSUMPTION_DIR']
RIDGE_CONSUMPTION_DIR = CONFIG['MODELS']['RIDGE_CONSUMPTION_DIR']
SCALER_DIR = CONFIG['MODELS']['SCALER_DIR']

CNN_FEATURE_SAVE_DIR = CONFIG['RESULTS']['CNN_FEATURE_SAVE_DIR']
RIDGE_PHONE_DENSITY_SAVE_DIR = CONFIG['RESULTS']['RIDGE_PHONE_DENSITY_SAVE_DIR']
RIDGE_PHONE_CONSUMPTION_SAVE_DIR = CONFIG['RESULTS']['RIDGE_PHONE_CONSUMPTION_SAVE_DIR']
RIDGE_CONSUMPTION_SAVE_DIR = CONFIG['RESULTS']['RIDGE_CONSUMPTION_SAVE_DIR']

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using {DEVICE} as backend...')

def create_folders():
    os.makedirs(CNN_FEATURE_SAVE_DIR, exist_ok=True)
    os.makedirs(RIDGE_PHONE_DENSITY_SAVE_DIR, exist_ok=True)
    os.makedirs(RIDGE_PHONE_CONSUMPTION_SAVE_DIR, exist_ok=True)
    os.makedirs(RIDGE_CONSUMPTION_SAVE_DIR, exist_ok=True)

def filename_to_im_tensor(file, transformer):
    im = plt.imread(file)[:,:,:3]
    im = transformer(im)
    return im[None].to(DEVICE)

class ModelPipeline:
    def __init__(self):
        print('loading CNN...')
        self.cnn = torch.load(CNN_DIR, map_location=DEVICE).eval()
        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        print('loading Ridge Regression models...')
        self.ridge_phone_density = joblib.load(RIDGE_PHONE_DENSITY_DIR)
        self.ridge_phone_consumption = joblib.load(RIDGE_PHONE_CONSUMPTION_DIR)
        self.ridge_consumption = joblib.load(RIDGE_CONSUMPTION_DIR)
        print()

    def run_pipeline(self, metric):
        assert metric in ['phone_density', 'phone_consumption', 'consumption']
        print(f'Running prediction pipeline on metric: {metric}')
        # check to see if clustered feats already exist
        SAVED_CLUSTER_FEATS_NAME = 'grid_features.npy'
        grids = None
        grid_features = None
        if SAVED_CLUSTER_FEATS_NAME in os.listdir(CNN_FEATURE_SAVE_DIR):
            print('Loading saved cluster features...')
            with open(os.path.join(CNN_FEATURE_SAVE_DIR, 'grid_names.pkl'), 'rb') as f:
                grids = pickle.load(f)
            grid_features = np.load(os.path.join(CNN_FEATURE_SAVE_DIR, SAVED_CLUSTER_FEATS_NAME))
        else:
            print('Reading reference dataframe...')
            try:
                df = pd.read_csv(os.path.join(GRID_DIR, 'image_download_locs.csv'))
            except Exception as e:
                logging.error('Make sure there is a file called image_download_locs.csv in ' + GRID_DIR, exc_info=True)
                exit(1)

            print('Extracting features using ' + IMAGE_DIR + ' as the image directory...')
            images, features = self.extract_features()

            print('Clustering the extracted features using the reference dataframe...')
            grids, grid_features = self.cluster_features(df, images, features, cluster_keys=['centroid_lat', 'centroid_lon'], image_key='image_name')

        print('Generating predictions using Ridge Regression model for given metric...')
        predictions = None
        SAVE_DIR = None
        if metric == 'phone_density':
            predictions = self.predict_phone_density(grid_features)
            SAVE_DIR = RIDGE_PHONE_DENSITY_SAVE_DIR

        elif metric == 'phone_consumption':
            predictions = self.predict_phone_consumption(grid_features)
            SAVE_DIR = RIDGE_PHONE_CONSUMPTION_SAVE_DIR

        elif metric == 'consumption':
            predictions = self.predict_consumption(grid_features)
            SAVE_DIR = RIDGE_CONSUMPTION_SAVE_DIR

        assert predictions is not None and SAVE_DIR is not None and len(grids) == len(predictions)

        print('Saving predictions to ' + os.path.join(SAVE_DIR, 'predictions.csv'))
        columns = ['centroid_lat', 'centroid_lon', f'predicted_{metric}_pc']
        with open(os.path.join(SAVE_DIR, 'predictions.csv'), 'w') as f:
            f.write(','.join(columns) + '\n')
            for (centroid_lat, centroid_lon), pred in zip(grids, predictions):
                to_write = [str(centroid_lat), str(centroid_lon), str(pred)]
                f.write(','.join(to_write) + '\n')

        print()


    def predict_nightlights(self):
        """
            Obtains nightlight predictions for all the images.
            
            Return: two items of equal length, one being the list of images and the other an array of shape (len(images), NUM_CLASSES)
        """
        SAVE_NAME = 'forward_classifications.npy'
        if SAVE_NAME in os.listdir(CNN_FEATURE_SAVE_DIR):
            print('Loading saved classifications...')
            ims = None
            with open(os.path.join(CNN_FEATURE_SAVE_DIR, 'image_names_classification.pkl'), 'rb') as f:
                ims = pickle.load(f)
            return ims, np.load(os.path.join(CNN_FEATURE_SAVE_DIR, SAVE_NAME))

        ims = os.listdir(IMAGE_DIR)
        path = os.path.join(IMAGE_DIR, '{}')

        i = 0
        batch_size = 4
        predictions = np.zeros((len(ims), 3))
#         pbar = tqdm(total=len(ims))
        pbar = CustomProgressBar(len(ims))

        # this approach uses batching and should offer a speed-up over passing one image at a time by nearly 10x
        # runtime should be 5-7 minutes vs 45+ for a full forward pass
        while i + batch_size < len(ims):
            ims_as_tensors = torch.cat([filename_to_im_tensor(path.format(ims[i+j]), self.transformer) for j in range(batch_size)], 0)
            predictions[i:i+batch_size,:] = self.cnn(ims_as_tensors).cpu().detach().numpy()
            i += batch_size
            pbar.update(batch_size)

        # does the final batch of remaining images
        if len(ims) - i != 0:
            rem = len(ims) - i
            ims_as_tensors = torch.cat([filename_to_im_tensor(path.format(ims[i+j]), self.transformer) for j in range(rem)], 0)
            predictions[i:i+rem,:] = self.cnn(ims_as_tensors).cpu().detach().numpy()
            i += rem
            pbar.update(rem)

        np.save(os.path.join(CNN_FEATURE_SAVE_DIR, SAVE_NAME), predictions)
        with open(os.path.join(CNN_FEATURE_SAVE_DIR, 'image_names_classification.pkl'), 'wb') as f:
            pickle.dump(ims, f)
        return ims, predictions

    def extract_features(self):
        """
            Obtains feature vectors for all the images.
            Saves results to disk for safekeeping as this can be a long step.
            
            Return: two items of equal length, one being the list of images and the other an array of shape (len(images), 4096)
        """
        SAVE_NAME = 'forward_features.npy'
        if SAVE_NAME in os.listdir(CNN_FEATURE_SAVE_DIR):
            print('Loading saved features...')
            ims = None
            with open(os.path.join(CNN_FEATURE_SAVE_DIR, 'image_names_feature_extraction.pkl'), 'rb') as f:
                ims = pickle.load(f)
            return ims, np.load(os.path.join(CNN_FEATURE_SAVE_DIR, SAVE_NAME))

        # we "rip" off the final layers so we can extract the 4096-size feature vector
        # this layer is the 4th on the classifier half of the CNN
        original = self.cnn.classifier
        ripped = self.cnn.classifier[:4]
        self.cnn.classifier = ripped

        ims = os.listdir(IMAGE_DIR)
        path = os.path.join(IMAGE_DIR, '{}')

        i = 0
        batch_size = 4
        features = np.zeros((len(ims), 4096))
        #  pbar = tqdm(total=len(ims))
        pbar = CustomProgressBar(len(ims))

        # this approach uses batching and should offer a speed-up over passing one image at a time by nearly 10x
        # runtime should be 8 minutes per 20k images on GPU
        print(f'Running forward pass on {len(ims)} images...')
        while i + batch_size < len(ims):
            ims_as_tensors = torch.cat([filename_to_im_tensor(path.format(ims[i+j]), self.transformer) for j in range(batch_size)], 0)
            features[i:i+batch_size,:] = self.cnn(ims_as_tensors).cpu().detach().numpy()
            i += batch_size
            if i % 100 == 0:
                pbar.update(100)

        # does the final batch of remaining images
        if len(ims) - i != 0:
            rem = len(ims) - i
            ims_as_tensors = torch.cat([filename_to_im_tensor(path.format(ims[i+j]), self.transformer) for j in range(rem)], 0)
            features[i:i+rem,:] = self.cnn(ims_as_tensors).cpu().detach().numpy()
            i += rem
            pbar.update(rem)
        
        print()
        self.cnn.classifier = original
        np.save(os.path.join(CNN_FEATURE_SAVE_DIR, SAVE_NAME), features)
        with open(os.path.join(CNN_FEATURE_SAVE_DIR, 'image_names_feature_extraction.pkl'), 'wb') as f:
            pickle.dump(ims, f)
        return ims, features

    def cluster_features(self, df, images, features, cluster_keys, image_key):
        """
            Aggregates the features based on a key(s) in the dataframe (specified by cluster_keys) against the image column in the dataframe (specified by image_key)
            - df: the dataframe
            - images: list of images that should have been outputted by extract_features
            - features: array of features created by extract_features
            - cluster_keys: key(s) to cluster on
            - image_key: key which has image names

            df[image_key] should not contain any images that are not in the list of image names

            Returns: two items of equal length, the first being a list of grids and the second being a cluster-aggregated feature array of shape (NUM_grids, 4096)
        """
        SAVE_NAME = 'grid_features.npy'
        if SAVE_NAME in os.listdir(CNN_FEATURE_SAVE_DIR):
            print('Loading saved features...')
            grids = None
            with open(os.path.join(CNN_FEATURE_SAVE_DIR, 'grid_names.pkl'), 'rb') as f:
                grids = pickle.load(f)
            return grids, np.load(os.path.join(CNN_FEATURE_SAVE_DIR, SAVE_NAME))
        
        assert len(images) == len(features)
        if type(cluster_keys) is not list:
            cluster_keys = [cluster_keys]

        df_lookup = pd.DataFrame.from_dict({image_key: images, 'feature_index': [i for i in range(len(images))]})
        prev_shape = len(df)
        df = pd.merge(df, df_lookup, on=image_key)
        assert prev_shape == len(df), print('The reference dataframe lookup did not merge with the images downloaded')

        grouped = df.groupby(cluster_keys)
        clustered_feats = np.zeros((len(grouped), 4096))
        grids = []
        for i, ((clust_lat, clust_lon), group) in enumerate(grouped):
            group_feats = np.zeros((len(group), 4096))
            for j, feat_idx in enumerate(group['feature_index']):
                group_feats[j,:] = features[feat_idx,:]
            group_feats = group_feats.mean(axis=0)
            clustered_feats[i,:] = group_feats
            grids.append([clust_lat, clust_lon])
        
        np.save(os.path.join(CNN_FEATURE_SAVE_DIR, SAVE_NAME), clustered_feats)
        with open(os.path.join(CNN_FEATURE_SAVE_DIR, 'grid_names.pkl'), 'wb') as f:
            pickle.dump(grids, f)
        return grids, clustered_feats

    def predict_phone_density(self, clustered_feats):
        return self.ridge_phone_density.predict(clustered_feats)

    def predict_phone_consumption(self, clustered_feats):
        return self.ridge_phone_consumption.predict(clustered_feats)

    def predict_consumption(self, clustered_feats):
        return self.ridge_consumption.predict(clustered_feats)



if __name__ == '__main__':
    create_folders()
    mp = ModelPipeline()

    arg = 'all'
    if len(sys.argv) >= 2:
        arg = sys.argv[1]
        assert arg in ['--all', '--extract-features', '--predict-consumption', '--predict-phone-consumption', '--predict-phone-density']

    if arg == '--extract-features':
        mp.extract_features()

    elif arg == '--predict-consumption':
        mp.run_pipeline(metric='consumption')

    elif arg == '--predict-phone-consumption':
        mp.run_pipeline(metric='phone_consumption')

    elif arg == '--predict-phone-density':
        mp.run_pipeline(metric='phone_density')

    elif arg == '--all':
        for metric in ['consumption', 'phone_density', 'phone_consumption']:
            mp.run_pipeline(metric=metric)
    
    else:
        raise ValueError('Args not handled correctly')

