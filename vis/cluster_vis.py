"""
Visualize cluster predictions

Written by Jatin Mathur.
5/2020
"""
import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
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
FIGURES_DIR = os.path.join(RESULTS_DIR, TYPE, COUNTRY, METRIC, 'figures')

assert TYPE in ['single_country', 'country_held_out']
assert COUNTRY in ['malawi_2016', 'ethiopia_2015']
assert METRIC in ['house_has_cellphone', 'est_monthly_phone_cost_pc']


def create_folders():
    os.makedirs(FIGURES_DIR, exist_ok=True)


def load_data():
    return pd.read_csv(os.path.join(RESULTS_DIR, TYPE, COUNTRY, METRIC, 'cluster_predictions', f'{METRIC}.csv'))


def plot_predictions():
    df_preds = load_data()
    df_valid = df_preds[~df_preds['is_train']]
    y = df_valid[METRIC].values
    yhat = df_valid[f'pred_{METRIC}'].values
    # probabilistic linear regression has this solution for
    # recovering the standard deviation after finding the predictions
    std = np.sqrt(((yhat - y)**2).sum() / len(yhat))
    # for the purposes of clean plotting, we won't show anything too far beyond 
    # the max of what our model predicts
    max_y = None if METRIC == 'house_has_cellphone' else max(yhat) + 2
    if max_y is not None:
        yhat = yhat[y < max_y]
        y = y[y < max_y]
    # these are ordered for the plotting
    y_i = np.unique(y)
    yhat_i = np.poly1d(np.polyfit(y, yhat, 1))(np.unique(y))
    r2 = r2_score(y, yhat)
    pi = 1 * std # prediction interval

    fig, ax = plt.subplots()
    ax.scatter(y, yhat, alpha=0.4)
    ax.plot(y_i, yhat_i)
    xloc = 0.75 * max(y_i)
    yloc = 0.75 * max(yhat_i)
    plt.text(xloc, yloc, f'r^2={round(r2, 2)}', size=12)
    ax.fill_between(y_i, (yhat_i - pi), (yhat_i + pi), color='b', alpha=.1)
    ax.set_xlabel('Observed Value')
    ax.set_ylabel('Predicted Value')
    ax.set_title(f'{COUNTRY} Observed vs Actual with Prediction Intervals (1 std)\nMetric: {METRIC}')

    savepath = os.path.join(FIGURES_DIR, f'{METRIC}.png')
    print(f'saving to {savepath}')
    fig.savefig(savepath)


if __name__ == '__main__':
    create_folders()
    plot_predictions()
