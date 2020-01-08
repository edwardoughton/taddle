"""
Visualization script

Written by Ed Oughton

January 2020

"""
import os
import configparser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import contextily as ctx
import geopandas as gpd
import matplotlib.colors

CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__), '..','scripts', 'script_config.ini'))
BASE_PATH = CONFIG['file_locations']['base_path']

DATA_RAW = os.path.join(BASE_PATH, 'raw')
DATA_PROCESSED = os.path.join(BASE_PATH, 'processed')
DATA_RESULTS = os.path.join(BASE_PATH, '..', 'results')


def create_regplot(data):

    # coeff_urban = round(coefficients['urban_{}'.format(distance)].values[0], 3)
    # coeff_rural = round(coefficients['rural_{}'.format(distance)].values[0], 3)


    # metric = "luminosity_sum_{}".format(distance)
    # metric2 = 'cons_pred_{}'.format(distance)
    # to_plot = clust_averages[[metric, "cons", metric2, "urban"]]

    # urban = to_plot.loc[to_plot['urban'] == 'urban']
    # rural = to_plot.loc[to_plot['urban'] == 'rural']

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, figsize=(10,10))

    g = sns.regplot(x="cons_pred_10km", y="cons", data=data, ax=ax1)
    g.set(xlabel='Predicted Consumption ($)', ylabel='Consumption ($)',
        title='Consumption vs \n Predicted Consumption (Nightlights)')

    g = sns.regplot(x="predicted_cons", y="cons", data=data, ax=ax2)
    g.set(xlabel='Predicted Consumption ($ per month)', ylabel='Consumption ($ per month)',
        title='Consumption vs \n Luminosity (CNN)')

    # g = sns.regplot(x="predicted_phone_cons", y="cluster_phone_cons", data=data, ax=ax3)
    # g.set(xlabel='Predicted Consumption ($)', ylabel='Consumption ($)',
    #     title='Consumption vs \n Predicted Consumption (Nightlights)')

    g = sns.regplot(x="predicted_phone_cons", y="cluster_phone_cons", data=data, ax=ax4)
    g.set(xlabel='Annual Spending ($)', ylabel='Predicted Annual Spending ($)',
        title='Predicted Spending on Phone Services (rexp_cat083) (CNN)')

    # g = sns.regplot(x="predicted_phone_cons", y="cluster_phone_cons", data=data, ax=ax3)
    # g.set(xlabel='Predicted Consumption ($)', ylabel='Consumption ($)',
    #     title='Consumption vs \n Predicted Consumption (Nightlights)')

    g = sns.regplot(x="predicted_cluster_hh_f35", y="cluster_hh_f35", data=data, ax=ax6)
    g.set(xlabel='Annual Cost ($)', ylabel='Annual Predicted Cost ($)',
        title='Predicted Annual Cost of Cell Phone Services (hh_f35) (CNN)')

    # g = sns.regplot(x="predicted_phone_cons", y="cluster_phone_cons", data=data, ax=ax3)
    # g.set(xlabel='Predicted Consumption ($)', ylabel='Consumption ($)',
    #     title='Consumption vs \n Predicted Consumption (Nightlights)')

    g = sns.regplot(x="predicted_cluster_hh_f34", y="cluster_hh_f34", data=data, ax=ax8)
    g.set(xlabel='Cell Phones per Household', ylabel='Predicted Cell Phones per Household',
        title='Predicted Cell Phones (hh_f34) (CNN)')

    fig.tight_layout()
    fig.savefig(os.path.join(BASE_PATH, '..', 'vis','figures', 'regplot.png'))

    return print('Completed regplot')


if __name__ == '__main__':

    data_1 = pd.read_csv(os.path.join(DATA_PROCESSED, 'lsms-cluster-2016.csv'))

    data_1 = data_1[pd.notnull(data_1['cons_pred_10km'])]

    data_2 =  pd.read_csv(os.path.join(DATA_PROCESSED, 'cluster_cnn_predictions.csv'))

    data = pd.merge(data_1, data_2, on=['lat', 'lon'])

    create_regplot(data)
