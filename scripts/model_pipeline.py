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
from utils import RidgeEnsemble

CONFIG = configparser.ConfigParser()
CONFIG.read('script_config.ini')

COUNTRY = CONFIG['DEFAULT']['COUNTRY']
GRID_DIR = f'countries/{COUNTRY}/grid'
IMAGE_DIR = f'countries/{COUNTRY}/images'

CNN_DIR = CONFIG['MODELS']['CNN_DIR']
RIDGE_PHONE_DENSITY_DIR = CONFIG['MODELS']['RIDGE_PHONE_DENSITY_DIR']
RIDGE_PHONE_CONSUMPTION_DIR = CONFIG['MODELS']['RIDGE_PHONE_CONSUMPTION_DIR']
RIDGE_CONSUMPTION_DIR = CONFIG['MODELS']['RIDGE_CONSUMPTION_DIR']

CNN_FEATURE_SAVE_DIR = f'countries/{COUNTRY}/results/cnn'
RIDGE_PHONE_DENSITY_SAVE_DIR = f'countries/{COUNTRY}/results/ridge_phone_density'
RIDGE_PHONE_CONSUMPTION_SAVE_DIR = f'countries/{COUNTRY}/results/ridge_phone_consumption'
RIDGE_CONSUMPTION_SAVE_DIR = f'countries/{COUNTRY}/results/ridge_consumption'

FORWARD_CLASSIFICATIONS = 'forward_classifications.npy'
IMAGE_NAMES_CLASSIFICATION = 'image_names_classification.pkl'

FORWARD_FEATURE_EXTRACT = 'forward_features.npy'
IMAGE_NAMES_FEATURE_EXTRACT = 'image_names_feature_extract.pkl'

GRID_FEATURES = 'grid_features.npy'
GRID_NAMES = 'grid_names.pkl'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using {DEVICE} as backend...')

def create_folders():
    os.makedirs(f'countries/{COUNTRY}/results', exist_ok=True)
    os.makedirs(CNN_FEATURE_SAVE_DIR, exist_ok=True)
    os.makedirs(RIDGE_PHONE_DENSITY_SAVE_DIR, exist_ok=True)
    os.makedirs(RIDGE_PHONE_CONSUMPTION_SAVE_DIR, exist_ok=True)
    os.makedirs(RIDGE_CONSUMPTION_SAVE_DIR, exist_ok=True)

# custom dataset for fast image loading and processing
class ForwardPassDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transformer):
        self.image_dir = image_dir
        self.image_list = os.listdir(self.image_dir)
        self.transformer = transformer

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = self.image_list[index]

        # Load image
        X = self.filename_to_im_tensor(self.image_dir + '/' + image_name)
        
        # dataloaders need to return a label, but for the forward pass we don't really care
        return X, -1
    
    def filename_to_im_tensor(self, file):
        im = plt.imread(file)[:,:,:3]
        im = self.transformer(im)
        return im

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
        grids = None
        grid_features = None
        if GRID_FEATURES in os.listdir(CNN_FEATURE_SAVE_DIR):
            print('Loading saved cluster features...')
            with open(os.path.join(CNN_FEATURE_SAVE_DIR, GRID_NAMES), 'rb') as f:
                grids = pickle.load(f)
            grid_features = np.load(os.path.join(CNN_FEATURE_SAVE_DIR, GRID_FEATURES))
        else:
            print('Reading reference dataframe...')
            try:
                df = pd.read_csv(os.path.join(GRID_DIR, 'image_download_locs.csv'))
            except Exception as e:
                logging.error('Make sure there is a file called image_download_locs.csv in ' + GRID_DIR, exc_info=True)
                exit(1)

            print('Extracting features using ' + IMAGE_DIR + ' as the image directory...')
            im_names, features = self.extract_features()

            print('Clustering the extracted features using the reference dataframe...')
            grids, grid_features = self.cluster_features(df, im_names, features, cluster_keys=['centroid_lat', 'centroid_lon'], image_key='image_name')

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
        if FORWARD_CLASSIFICATIONS in os.listdir(CNN_FEATURE_SAVE_DIR):
            print('Loading saved classifications...')
            im_names = None
            with open(os.path.join(CNN_FEATURE_SAVE_DIR, IMAGE_NAMES_CLASSIFICATION), 'rb') as f:
                im_names = pickle.load(f)
            return im_names, np.load(os.path.join(CNN_FEATURE_SAVE_DIR, FORWARD_CLASSIFICATIONS))

        im_names = os.listdir(IMAGE_DIR)
        path = os.path.join(IMAGE_DIR, '{}')

        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        batch_size = 8
        num_workers = 4
        
        print('Initializing dataset and dataloader...')
        dataset = ForwardPassDataset(IMAGE_DIR, transformer)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        image_order = dataset.image_list

        predictions = np.zeros((len(image_order), 1))

        # this approach uses batching and should offer a speed-up over passing one image at a time by nearly 10x
        # runtime should be 5 minutes per 20k images on GPU
        print(f'Running predictions on {len(image_order)} images...')
        i = 0
        for inputs, _ in tqdm(dataloader):
            inputs = inputs.to(DEVICE)
            outputs = self.cnn(inputs)
            predictions[i:i+batch_size,:] = outputs.cpu().detach().numpy()
            i += len(inputs)

        np.save(os.path.join(CNN_FEATURE_SAVE_DIR, FORWARD_CLASSIFICATIONS), predictions)
        with open(os.path.join(CNN_FEATURE_SAVE_DIR, IMAGE_NAMES_CLASSIFICATION), 'wb') as f:
            pickle.dump(image_order, f)
        return image_order, predictions

    def extract_features(self):
        """
            Obtains feature vectors for all the images.
            Saves results to disk for safekeeping as this can be a long step.

            Return: two items of equal length, one being the list of images and the other an array of shape (len(images), 4096)
        """
        if FORWARD_FEATURE_EXTRACT in os.listdir(CNN_FEATURE_SAVE_DIR):
            print('Loading saved features...')
            im_names = None
            with open(os.path.join(CNN_FEATURE_SAVE_DIR, IMAGE_NAMES_FEATURE_EXTRACT), 'rb') as f:
                im_names = pickle.load(f)
            return im_names, np.load(os.path.join(CNN_FEATURE_SAVE_DIR, FORWARD_FEATURE_EXTRACT))

        # we "rip" off the final layers so we can extract the 4096-size feature vector
        # this layer is the 4th on the classifier half of the CNN
        original = self.cnn.classifier
        ripped = self.cnn.classifier[:4]
        self.cnn.classifier = ripped
        
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        batch_size = 8
        num_workers = 4
        
        print('Initializing dataset and dataloader...')
        dataset = ForwardPassDataset(IMAGE_DIR, transformer)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        image_order = dataset.image_list
        
        features = np.zeros((len(image_order), 4096))

        # this approach uses batching and should offer a speed-up over passing one image at a time by nearly 10x
        # runtime should be 5 minutes per 20k images on GPU
        print(f'Running forward pass on {len(image_order)} images...')
        i = 0
        for inputs, _ in tqdm(dataloader):
            inputs = inputs.to(DEVICE)
            outputs = self.cnn(inputs)
            features[i:i+len(inputs),:] = outputs.cpu().detach().numpy()
            i += len(inputs)

        print()
        self.cnn.classifier = original
        np.save(os.path.join(CNN_FEATURE_SAVE_DIR, FORWARD_FEATURE_EXTRACT), features)
        with open(os.path.join(CNN_FEATURE_SAVE_DIR, IMAGE_NAMES_FEATURE_EXTRACT), 'wb') as f:
            pickle.dump(image_order, f)
        return image_order, features

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
        if GRID_FEATURES in os.listdir(CNN_FEATURE_SAVE_DIR):
            print('Loading saved features...')
            grids = None
            with open(os.path.join(CNN_FEATURE_SAVE_DIR, GRID_NAMES), 'rb') as f:
                grids = pickle.load(f)
            return grids, np.load(os.path.join(CNN_FEATURE_SAVE_DIR, GRID_FEATURES))

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

        np.save(os.path.join(CNN_FEATURE_SAVE_DIR, GRID_FEATURES), clustered_feats)
        with open(os.path.join(CNN_FEATURE_SAVE_DIR, GRID_NAMES), 'wb') as f:
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

    arg = '--all'
    if len(sys.argv) >= 2:
        arg = sys.argv[1]
        assert arg in ['--all', '--extract-features', '--predict-consumption', '--predict-phone-consumption', '--predict-phone-density']
        
    if arg == '--extract-features':
        mp.extract_features()
        exit(0)

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
