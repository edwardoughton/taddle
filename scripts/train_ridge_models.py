"""
Create Ridge Regression models using CNN features to predict metrics.
Run after train_cnn.ipynb and feature_extract.ipynb

Written by Jatin Mathur.
5/2020
"""

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import joblib

BASE_DIR = '.'
# repo imports
import sys
sys.path.append(BASE_DIR)
from utils import merge_on_lat_lon, assign_groups, run_randomized_cv, run_spatial_cv, RidgeEnsemble, train_and_predict_ridge
from config import TRAINING_CONFIG, RANDOM_SEED

COUNTRIES_DIR = os.path.join(BASE_DIR, 'data', 'countries')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

TYPE = TRAINING_CONFIG['TYPE']
COUNTRY = TRAINING_CONFIG['COUNTRY']
METRIC = TRAINING_CONFIG['METRIC']

assert TYPE in ['single_country', 'country_held_out']
assert COUNTRY in ['malawi_2016', 'ethiopia_2015']
assert METRIC in ['house_has_cellphone', 'est_monthly_phone_cost_pc']

def load_country(typ, country, metric):
    '''
    Organizes the country's dataframe so that each index corresponds to the index in the cluster features
    Returns the cluster features and the organized dataframe
    '''
    country_results_dir = os.path.join(RESULTS_DIR, typ, country, metric, 'cnn')
    df_clusters = None
    if typ == 'country_held_out':
        # df_clusters will be the concatenation of each country's dataframe
        df_mw = pd.read_csv(os.path.join(COUNTRIES_DIR, 'malawi_2016', 'processed', 'clusters.csv'))
        df_eth = pd.read_csv(os.path.join(COUNTRIES_DIR, 'ethiopia_2015', 'processed', 'clusters.csv'))
        df_clusters = pd.concat([df_mw, df_eth])
    else:
        country_processed_dir = os.path.join(COUNTRIES_DIR, country, 'processed')
        df_clusters = pd.read_csv(os.path.join(country_processed_dir, 'clusters.csv'))
    x_train = np.load(os.path.join(country_results_dir, f'cluster_feats_train_{metric}.npy'))
    cluster_train = pickle.load(open(os.path.join(country_results_dir, f'cluster_order_train_{metric}.pkl'), 'rb'))
    cluster_train = pd.DataFrame.from_records(cluster_train, columns=['cluster_lat', 'cluster_lon'])
    cluster_train['feat_index'] = np.arange(len(cluster_train))
    df_train = merge_on_lat_lon(df_clusters, cluster_train, keys=['cluster_lat', 'cluster_lon'])
    df_train.sort_values('feat_index', ascending=True, inplace=True)
    
    x_valid = np.load(os.path.join(country_results_dir, f'cluster_feats_valid_{metric}.npy'))
    cluster_valid = pickle.load(open(os.path.join(country_results_dir, f'cluster_order_valid_{metric}.pkl'), 'rb'))
    cluster_valid = pd.DataFrame.from_records(cluster_valid, columns=['cluster_lat', 'cluster_lon'])
    cluster_valid['feat_index'] = np.arange(len(cluster_valid))
    df_valid = merge_on_lat_lon(df_clusters, cluster_valid, keys=['cluster_lat', 'cluster_lon'])
    df_valid.sort_values('feat_index', ascending=True, inplace=True)
    
    assert len(cluster_train) + len(cluster_valid) == len(df_clusters)
    assert len(df_train) + len(df_valid) == len(df_clusters)
    return x_train, df_train, x_valid, df_valid


if __name__ == '__main__':
    x_train, df_train, x_valid, df_valid = load_country(TYPE, COUNTRY, METRIC)
    y_train = df_train[METRIC].values
    y_valid = df_valid[METRIC].values
    
    # the variable names below are intended to identify them
    # for example: r2_rcv means r2 using randomized cross validation
    # another example: yhat_scv_train means yhat using spatial cross validation on the train set
    print("running randomized cv...")
    r2_rcv, _, ridges_rcv, scalers_rcv = run_randomized_cv(x_train, y_train, random_seed=RANDOM_SEED)
    re_rcv = RidgeEnsemble(ridges_rcv, scalers_rcv)
    yhat_rcv_valid = re_rcv.predict(x_valid)
    r2_rcv_valid = r2_score(y_valid, yhat_rcv_valid)
    print(f"randomized cv r2: {r2_rcv}, validation r2: {r2_rcv_valid}")
    pearson_r2_rcv = pearsonr(y_valid, yhat_rcv_valid)[0]**2
    print(f"validation pearson R squared: {pearson_r2_rcv}")

    print()
    
    groups, _ = assign_groups(df_train, 5, random_seed=RANDOM_SEED)
    print("running spatial cv...")
    r2_scv, yhat_scv_train, ridges_scv, scalers_scv = run_spatial_cv(x_train, y_train, groups, random_seed=RANDOM_SEED)
    re_scv = RidgeEnsemble(ridges_scv, scalers_scv)
    yhat_scv_valid = re_scv.predict(x_valid)
    r2_scv_valid = r2_score(y_valid, yhat_scv_valid)
    print(f"spatial cv r2: {r2_scv}, validation r2: {r2_scv_valid}")
    pearson_r2_scv = pearsonr(y_valid, yhat_scv_valid)[0]**2
    print(f"validation pearson R squared: {pearson_r2_scv}")

    print()
    savedir = os.path.join(RESULTS_DIR, TYPE, COUNTRY, METRIC, 'ridge_models')
    savepath = os.path.join(savedir, f'{METRIC}.joblib')
    os.makedirs(savedir, exist_ok=True)
    print(f"saving spatial-cv based models to {savepath}")
    joblib.dump(re_scv, savepath)

    savedir = os.path.join(RESULTS_DIR, TYPE, COUNTRY, METRIC, 'cluster_predictions')
    savepath = os.path.join(savedir, f'{METRIC}.csv')
    os.makedirs(savedir, exist_ok=True)
    print(f"saving cluster predictions to {savepath}")
    df_train[f'pred_{METRIC}'] = yhat_scv_train
    df_train['is_train'] = True
    df_valid[f'pred_{METRIC}'] = yhat_scv_valid
    df_valid['is_train'] = False
    df_clusters = pd.concat([df_train, df_valid])
    # keep only relavant columns for a concise, clear CSV
    df_clusters = df_clusters[['cluster_lat', 'cluster_lon', METRIC, f'pred_{METRIC}', 'is_train']]
    df_clusters.to_csv(savepath, index=False)
    