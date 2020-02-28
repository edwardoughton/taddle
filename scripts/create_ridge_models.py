"""
Create Ridge Regression models using LSMS survey data.

Written by Jatin Mathur

Winter 2020

"""

import configparser
import pandas as pd
import numpy as np
import math
import geoio
import os
import numpy as np
import pandas as pd
import random
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import sklearn.linear_model as linear_model
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
import seaborn as sns
import joblib
from sklearn import metrics
import pickle

# repo imports
import sys
sys.path.append('.')
from utils import RidgeEnsemble, merge_on_lat_lon

import warnings
warnings.filterwarnings('ignore')

CONFIG = configparser.ConfigParser()
CONFIG.read('script_config.ini')

COUNTRY = CONFIG['DEFAULT']['COUNTRY']

LSMS_DIR = f'data/LSMS/{COUNTRY}/input'
NIGHTLIGHTS_DIR = 'data/Nightlights/2013'
CLUSTER_DATA_DIR = f'data/LSMS/{COUNTRY}/processed/cluster_data.csv'
CLUSTER_PREDICTIONS_DIR = f'data/LSMS/{COUNTRY}/output/cluster_predictions.csv'

# Purchasing Power Adjustment
PPP = float(CONFIG['DEFAULT']['PPP'])

CNN_LSMS_CLUSTER_FEATS = f'data/LSMS/{COUNTRY}/processed/cluster_feats.npy'
CNN_LSMS_CLUSTER_ORDER = f'data/LSMS/{COUNTRY}/processed/cluster_order.pkl'


def create_folders():
    """
    Function to create new folder.

    """
    os.makedirs(f'data/LSMS/{COUNTRY}/processed', exist_ok=True)
    os.makedirs(f'data/LSMS/{COUNTRY}/output', exist_ok=True)


def create_space(lat, lon):
    """
    Creates a 100km^2 area bounding box.

    Parameters
    ----------
    lat : float
        Latitude
    lon : float
        Longitude

    """
    v = (180 / math.pi) * (5000 / 6378137) # approximately 0.045
    bottom = lat - v
    left = lon - v
    top = lat + v
    right = lon + v

    return bottom, left, top, right


def prepare_data():
    """
    Preprocessing script that prepares the data to be passed to the ridge models.

    """
    if os.path.exists(CLUSTER_DATA_DIR):
        print('data has already been pre-processed...')
        return

    print('preprocessing...')
    df = pd.read_stata(os.path.join(LSMS_DIR, 'IHS4 Consumption Aggregate.dta'))
    df['persons_in_household'] = df['rexpagg'] / df['rexpaggpc']
    df['annual_consumption_hh'] = df['rexpagg']

    # accounting for purchasing power parity
    df['annual_consumption_hh'] /= PPP
    df['annual_phone_consumption_hh'] = df['rexp_cat083']
    df['annual_phone_consumption_hh'] = df['annual_phone_consumption_hh'] / PPP

    # grab these columns
    df = df[[
        'case_id', 'annual_consumption_hh',
        'annual_phone_consumption_hh', 'persons_in_household'
        ]]

    path = os.path.join('HouseholdGeovariables_stata11', 'HouseholdGeovariablesIHS4.dta')
    df_geo = pd.read_stata(os.path.join(LSMS_DIR, path))

    df_cords = df_geo[['case_id', 'HHID', 'lat_modified', 'lon_modified']]
    df_cords.rename(columns={'lat_modified': 'lat', 'lon_modified': 'lon'}, inplace=True)

    df_hhf = pd.read_stata(os.path.join(LSMS_DIR, 'HH_MOD_F.dta'))
    df_hhf = df_hhf[['case_id', 'HHID', 'hh_f34', 'hh_f35']]

    df_hhf.rename(
        columns={'hh_f34': 'cellphones_ph',
        'hh_f35': 'estimated_annual_phone_cost_ph'},
        inplace=True)

    df = pd.merge(df, df_cords[['case_id', 'HHID']], on='case_id')
    df_combined = pd.merge(df, df_cords, on=['case_id', 'HHID'])
    df_combined = pd.merge(df_combined, df_hhf, on=['case_id', 'HHID'])

    warning = 'Have a household with null people'
    assert df_combined['persons_in_household'].isna().sum() == 0, print(warning)

    df_clusters = df_combined.groupby(['lat', 'lon']).sum().reset_index()

    # at this moment the data is per household
    data_cols = [
        'annual_consumption_hh', 'annual_phone_consumption_hh',
        'cellphones_ph', 'estimated_annual_phone_cost_ph'
    ]

    for c in data_cols:
        # persons in household is now really all persons surveyed in cluster
        # we can get the per capita value by dividing
        df_clusters[c[:-3] + '_pc'] = (
            df_clusters[c] / df_clusters['persons_in_household']
        )

    df_clusters.drop(data_cols, axis=1, inplace=True)
    df_clusters.rename(
        columns={'persons_in_household': 'persons_surveyed'},
        inplace=True)

    rename = {c: 'cluster_' + c for c in df_clusters.columns}
    df_clusters.rename(columns=rename, inplace=True)

    filename = 'F182013.v4c_web.stable_lights.avg_vis.tif'
    img = geoio.GeoImage(os.path.join(NIGHTLIGHTS_DIR, filename))
    im_array = np.squeeze(img.get_data())

    cluster_nightlights = []
    for _,r in df_clusters.iterrows():
        min_lat, min_lon, max_lat, max_lon = create_space(r.cluster_lat, r.cluster_lon)
        xminPixel, yminPixel = img.proj_to_raster(min_lon, min_lat)
        xmaxPixel, ymaxPixel = img.proj_to_raster(max_lon, max_lat)

        xminPixel, xmaxPixel = min(xminPixel, xmaxPixel), max(xminPixel, xmaxPixel)
        yminPixel, ymaxPixel = min(yminPixel, ymaxPixel), max(yminPixel, ymaxPixel)

        xminPixel, yminPixel, xmaxPixel, ymaxPixel = (
            int(xminPixel), int(yminPixel), int(xmaxPixel), int(ymaxPixel)
        )
        cluster_nightlights.append(
            im_array[yminPixel:ymaxPixel,xminPixel:xmaxPixel].mean())

    df_clusters['cluster_nightlights'] = cluster_nightlights

    # save preprocessed csv
    df_clusters.to_csv(CLUSTER_DATA_DIR, index=False)
    print(f'Saved preprocessing to {CLUSTER_DATA_DIR}')


class CreateRidge:
    """
    Class object to create ridge models.

    """
    def __init__(self):
        """
        Code has been adapted from Jean et al. 2016.

        """
        self.cluster_feats = np.load(CNN_LSMS_CLUSTER_FEATS)
        self.cluster_order = pickle.load(open(CNN_LSMS_CLUSTER_ORDER, 'rb'))
        self.cluster_data = pd.read_csv(CLUSTER_DATA_DIR)

        # makes sure the cluster order matches the cluster data one-to-one
        tmp = pd.DataFrame()
        tmp['cluster_lat'] = [x[0] for x in self.cluster_order]
        tmp['cluster_lon'] = [x[1] for x in self.cluster_order]
        tmp['feat_index'] = np.arange(len(tmp))
        res = merge_on_lat_lon(self.cluster_data, tmp)

        warning = 'Cluster order does not match preprocessed clusters.csv'
        assert len(res) == len(self.cluster_data), print(warning)
        self.cluster_data = res.sort_values(
            'feat_index', ascending=True).drop('feat_index', axis=1)

        self.cluster_results = self.cluster_data.copy()
        self.metric_to_preprocessed_name = {
            'consumption': 'cluster_annual_consumption_pc',
            'phone_consumption': 'cluster_annual_phone_consumption_pc',
            'phone_density': 'cluster_cellphones_pc'
        }


    def train_all(self):
        """
        Method to train across all parameters.

        """
        for metric in ['consumption', 'phone_consumption', 'phone_density']:
            self.train(metric)
        print(f'Saving predictions to {CLUSTER_PREDICTIONS_DIR}')
        self.cluster_results.to_csv(CLUSTER_PREDICTIONS_DIR, index=False)


    def train(self, metric):
        """
        Method to train across specific parameter.

        Parameter
        ---------
        metric : string
            Parameter to train with.

        """
        assert metric in ['consumption', 'phone_consumption', 'phone_density']
        col_name = self.metric_to_preprocessed_name[metric]
        y = self.cluster_data[col_name].values
        # adds this small value to prevent error when the value is 0
        y_log = np.log(y + 0.0001)

        print(f'Training model on log {metric}...')
        y_hat_log, r2, _, _ = self.predict_metric(self.cluster_feats, y_log)
        print(f'R2: {r2}')
        print()

        print(f'Training model on {metric}...')
        y_hat, r2, ridges, scalers = self.predict_metric(self.cluster_feats, y)
        print(f'R2: {r2}')
        print()

        re = RidgeEnsemble(ridges, scalers)
        r2 = metrics.r2_score(y, re.predict(self.cluster_feats))
        print(f'Ridge ensemble scores: {r2}...')
        ridge_ensemble_save_dir = f'models/ridge_{metric}.joblib'
        print(f'Saving ridge ensemble to {ridge_ensemble_save_dir}')
        joblib.dump(re, ridge_ensemble_save_dir)
        print()

        self.cluster_results[f'predicted_{metric}'] = y_hat
        self.cluster_results[f'predicted_log_{metric}'] = y_hat_log


    def predict_metric(self, X, y, k=5, k_inner=5, points=10, alpha_low=1,
        alpha_high=5, margin=0.25):
        """
        Predict metric method.

        Parameters
        ----------
        X : float

        y : float

        k : int

        k_inner : int

        points : int

        alpha_low : int

        alpha_high : int

        margin : float

        Returns
        -------
        y_hat : float

        r2 : float

        ridges :

        scalers :

        """
        y_hat, r2, ridges, scalers = self.run_cv(
            X, y, k, k_inner, points, alpha_low, alpha_high)

        return y_hat, r2, ridges, scalers


    def run_cv(self, X, y, k, k_inner, points, alpha_low, alpha_high,
        randomize=False):
        """
        Runs nested cross-validation to make predictions and compute
        r-squared.

        Parameters
        ----------
        X : float

        y : float

        k : int

        k_inner : int

        points : int

        alpha_low : int

        alpha_high : int

        randomize : string

        Returns
        -------
        y_hat : float

        r2.mean() : float

        ridges :

        scalers :

        """
        alphas = np.logspace(alpha_low, alpha_high, points)
        r2s = np.zeros((k,))
        y_hat = np.zeros_like(y)
        kf = KFold(n_splits=k, shuffle=True)
        fold = 0

        ridges = []
        scalers = []

        for train_idx, test_idx in kf.split(X):

            r2s, y_hat, fold, ridge, scaler = self.evaluate_fold(
                X, y, train_idx, test_idx, k_inner, alphas, r2s,
                y_hat, fold, randomize)

            ridges.append(ridge)
            scalers.append(scaler)

        return y_hat, r2s.mean(), ridges, scalers


    def evaluate_fold(self, X, y, train_idx, test_idx, k_inner,
        alphas, r2s, y_hat, fold, randomize):
        """
        Evaluates one fold of outer CV.

        Parameters
        ----------
        X : float

        y : float

        train_idx :

        test_idx :

        k_inner : int

        alphas :

        r2s :

        y_hat : float

        fold :

        randomize :

        Returns
        -------
        r2s : float

        y_hat : float

        fold + 1 : float

        ridges :

        scalers :

        """
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if randomize:
            random.shuffle(y_train)

        best_alpha = self.find_best_alpha(X_train, y_train, k_inner, alphas)
        X_train, X_test, scaler = self.scale_features(X_train, X_test)
        y_test_hat, ridge = self.train_and_predict_ridge(
            best_alpha, X_train, y_train, X_test)
        r2 = stats.pearsonr(y_test, y_test_hat)[0] ** 2
        r2s[fold] = r2
        y_hat[test_idx] = y_test_hat

        return r2s, y_hat, fold + 1, ridge, scaler


    def scale_features(self, X_train, X_test):
        """
        Scales features using StandardScaler.

        Parameters
        ----------
        X_train : float

        X_test : float

        Return
        ------
        X_train : float

        X_test : float

        X_scaler : float

        """
        X_scaler = StandardScaler(with_mean=True, with_std=False)
        X_train = X_scaler.fit_transform(X_train)
        X_test = X_scaler.transform(X_test)

        return X_train, X_test, X_scaler


    def train_and_predict_ridge(self, alpha, X_train, y_train, X_test):
        """
        Trains ridge model and predicts test set.

        Parameters
        ----------
        alpha : float

        X_train : float

        y_train : float

        X_test : float

        Return
        ------
        y_hat : float

        ridge :

        """
        ridge = linear_model.Ridge(alpha)
        ridge.fit(X_train, y_train)
        y_hat = ridge.predict(X_test)

        return y_hat, ridge


    def predict_inner_test_fold(self, X, y, y_hat, train_idx, test_idx, alpha):
        """
        Predicts inner test fold.

        Parameters
        ----------
        X : float

        y : float

        y_hat : float

        train_idx :

        test_idx :

        alphas :

        Returns
        -------
        y_hat : float

        """
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        X_train, X_test, _ = self.scale_features(X_train, X_test)
        y_hat[test_idx], _ = self.train_and_predict_ridge(
            alpha, X_train, y_train, X_test)

        return y_hat


    def find_best_alpha(self, X, y, k_inner, alphas):
        """
        Finds the best alpha in an inner CV loop.

        Parameters
        ----------
        X : float

        y : float

        k_inner :

        alphas :

        Return
        ------
        best_alpha : float

        """
        kf = KFold(n_splits=k_inner, shuffle=True)
        best_alpha = 0
        best_r2 = 0

        for idx, alpha in enumerate(alphas):

            y_hat = np.zeros_like(y)

            for train_idx, test_idx in kf.split(X):
                y_hat = self.predict_inner_test_fold(
                    X, y, y_hat, train_idx, test_idx, alpha)
            r2 = stats.pearsonr(y, y_hat)[0] ** 2

            if r2 > best_r2:
                best_alpha = alpha
                best_r2 = r2

        return best_alpha


if __name__ == '__main__':

    create_folders()

    arg = '--all'
    if len(sys.argv) >= 2:
        arg = sys.argv[1]
        assert arg in [
            '--all', '--preprocess', '--train', '--train-consumption',
            '--train-phone-consumption', '--train-phone-density'
        ]

    prepare_data()
    if arg == '--preprocess':
        exit(0)

    warning = (f'Make sure you have applied the CNN on the cluster images \
        and saved the result to {CNN_LSMS_CLUSTER_FEATS}`')

    assert os.path.isfile(CNN_LSMS_CLUSTER_FEATS), print(warning)

    if arg == '--all' or '--train' in arg:
        cr = CreateRidge()

        if arg in ['--all', '--train']:
            cr.train_all()
        elif arg == '--train-consumption':
            cr.train('consumption')
        elif arg == '--train-phone-consumption':
            cr.train('phone_consumption')
        elif arg == '--train-phone-density':
            cr.train('phone_density')
        else:
            raise ValueError('Args not handled correctly')
