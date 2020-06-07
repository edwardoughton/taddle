"""
For a new country's images:
1) extract features using CNN
2) aggregate cluster-level features
3) use ridge models to predict device penetration ("house has cellphone") and monthly cost of phone services ("est_monthly_phone_cost_pc")

Written by Jatin Mathur
5/2020
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import pickle
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from PIL import Image
import time
import copy

BASE_DIR = '.'
import sys
sys.path.append(BASE_DIR)
from utils import merge_on_lat_lon
from config import PREDICTION_MAPS_CONFIG, RANDOM_SEED

COUNTRIES_DIR = os.path.join(BASE_DIR, 'data', 'countries')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

COUNTRY_ABBRV = PREDICTION_MAPS_CONFIG['COUNTRY_ABBRV']
CNN_GRID_OUTPUTS = os.path.join(RESULTS_DIR, 'prediction_maps', COUNTRY_ABBRV, 'cnn')
GRID_DIR = os.path.join(COUNTRIES_DIR, COUNTRY_ABBRV, 'grid')
IMAGE_DIR = os.path.join(COUNTRIES_DIR, COUNTRY_ABBRV, 'images')

TYPE = PREDICTION_MAPS_CONFIG['TYPE']
COUNTRY = PREDICTION_MAPS_CONFIG['COUNTRY']
METRIC = PREDICTION_MAPS_CONFIG['METRIC']

CNN_SAVE_DIR = os.path.join(BASE_DIR, 'models', TYPE, COUNTRY, METRIC)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {DEVICE} as backend')

assert TYPE in ['single_country', 'country_held_out']
assert COUNTRY in ['malawi_2016', 'ethiopia_2015']
assert METRIC in ['house_has_cellphone', 'est_monthly_phone_cost_pc']


def create_folders():
    os.makedirs(CNN_GRID_OUTPUTS, exist_ok=True)
    return


def load_country_abbrv_df():
    print('loading images dataframe')
    filepath = os.path.join(GRID_DIR, 'image_download_locs.csv')
    df_images = pd.read_csv(filepath)
    return df_images


def load_model():
    print('loading model')
    model = torch.load(os.path.join(CNN_SAVE_DIR, f'trained_model_{METRIC}.pt'), map_location=DEVICE)
    return model


def modify_model(model):
    '''
    Makes feature extraction possible by removing all layers afterwards
    A forward pass through the model will return the values at this layer
    instead of the predicted class
    '''
    print('modifying model for feature extraction')
    model.classifier = model.classifier[:4]


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
        
        # dataloaders need to return a label, but for the feature extract we don't really care
        return X, -1
    
    def filename_to_im_tensor(self, file):
        im = (plt.imread(file)[:,:,:3] * 256).astype(np.uint8)
        im = Image.fromarray(im)
        im = self.transformer(im)
        return im
    

def run_forward_pass(model, df_images):
    input_size = 224
    transformer = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model.eval()  
    # shape of final array will be (num_images, 4096)
    # we also want to record the image each index represents
    feats = np.zeros((len(os.listdir(IMAGE_DIR)), 4096))
    image_order = []
    i = 0
    # use the images to do the forward pass
    dataset = ForwardPassDataset(IMAGE_DIR, transformer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
    image_order += dataset.image_list
    # forward pass for this class
    print('running forward pass')
    for inputs, _ in tqdm(dataloader):
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        feats[i:i+len(inputs),:] = outputs.cpu().detach().numpy()
        i += len(inputs)
            
    forward_pass_df = pd.DataFrame.from_dict({'image_name': image_order, 'feat_index': np.arange(len(image_order))})
    return feats, forward_pass_df


def aggregate_features(df, feats):
    '''
    Aggregates features feats using a dataframe df with columns 'centroid_lat', 'centroid_lon', 'feat_index'
    '''
    group = df.groupby(['centroid_lat', 'centroid_lon'])
    x = np.zeros((len(group), 4096))
    cluster_list = [] # the corresponding clusters (lat, lon) to the x aggregate feature array
    for i, g in enumerate(group):
        lat, lon = g[0]
        im_sub = df[(df['centroid_lat'] == lat) & (df['centroid_lon'] == lon)].reset_index(drop=True)
        agg_feats = np.zeros((len(im_sub), 4096))
        for j, d in im_sub.iterrows():
            agg_feats[j,:] = feats[d.feat_index]
        agg_feats = agg_feats.mean(axis=0) # averages the features across all images in the cluster

        x[i,:] = agg_feats
        cluster_list.append([lat, lon])
        
    # save to the correct directory
    print(f'saving to {CNN_GRID_OUTPUTS}')
    np.save(os.path.join(CNN_GRID_OUTPUTS, f'centroid_feats_{METRIC}.npy'), x)
    pickle.dump(cluster_list, open(os.path.join(CNN_GRID_OUTPUTS, f'centroid_order_{METRIC}.pkl'), 'wb'))


def predict_grid_values(feats, grid_list):
    '''
    Creates predictions at the grid level using
    feats: a numpy array of features at the grid level
    grid_list: a 2d list of corresponding centroid lats and lons for feats

    grid_list and feats should line up one-to-one
    '''
    grid_list = pd.DataFrame.from_records(grid_list, columns=['centroid_lat', 'centroid_lon'])
    ridgepath = os.path.join(RESULTS_DIR, TYPE, COUNTRY, METRIC, 'ridge_models', f'{METRIC}.joblib')
    re = joblib.load(ridgepath)
    grid_list[f'pred_{METRIC}'] = re.predict(feats)
    filepath = os.path.join(CNN_GRID_OUTPUTS, f'pred_{METRIC}.csv')
    print(f'saving predictions to {filepath}')
    grid_list.to_csv(filepath, index=False)


if __name__ == '__main__':
    create_folders()
    feats_path = os.path.join(CNN_GRID_OUTPUTS, f'centroid_feats_{METRIC}.npy')
    grid_list_path = os.path.join(CNN_GRID_OUTPUTS, f'centroid_order_{METRIC}.pkl')

    if not os.path.exists(feats_path):
        print('running forward pass')
        df_images = load_country_abbrv_df()
        model = load_model()
        modify_model(model)
        feats, df_forward_pass = run_forward_pass(model, df_images)
        df = pd.merge(left=df_images, right=df_forward_pass, on='image_name', how='inner')
        aggregate_features(df, feats)

    print('forward pass and aggregation are complete')
    feats = np.load(feats_path)
    grid_list = pickle.load(open(grid_list_path, 'rb'))
    predict_grid_values(feats, grid_list)

