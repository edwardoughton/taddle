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
import geopandas as gpd
import matplotlib.colors

CONFIG_DATA = configparser.ConfigParser()
CONFIG_DATA.read(os.path.join(os.path.dirname(__file__), '..', 'scripts','script_config.ini'))
BASE_PATH = CONFIG_DATA['file_locations']['base_path']

CONFIG_COUNTRY = configparser.ConfigParser()
CONFIG_COUNTRY.read('script_config.ini')
COUNTRY = CONFIG_COUNTRY['DEFAULT']['COUNTRY']
SHAPEFILE_DIR = f'countries/{COUNTRY}/shapefile'
GRID_DIR = f'countries/{COUNTRY}/grid'


def create_folders():

    path = os.path.join(BASE_PATH, '..', 'vis','figures')
    if not os.path.exists(path):
        os.mkdir(path)


def r2(x, y):
    coef = round(np.corrcoef(x, y)[0, 1]**2, 3)
    return coef


def plot(x, y, x_label, y_label):

    g = sns.jointplot(x, y, stat_func=r2,
        kind="reg").set_axis_labels(x_label, y_label)
    g.savefig(os.path.join(BASE_PATH, '..', 'vis','figures', 'jointplot_{}_{}.png'.format(x_label, y_label)))


def create_regplot(data):

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,8))

    coef1 = r2(data['nightlights'], data['cons'])
    g = sns.regplot(x="nightlights", y="cons", data=data, ax=ax1)
    g.set(xlabel='Luminosity (DN)', ylabel='Consumption ($ per month)',
        title='Luminosity vs\nConsumption (R$^2$={})'.format(str(coef1)))

    coef2 = r2(data['nightlights'], data['cluster_phone_cons'])
    g = sns.regplot(x="nightlights", y="cluster_phone_cons", data=data, ax=ax2)
    g.set(xlabel='Luminosity (DN)', ylabel='Total Cost ($ per month)',
        title='Luminosity vs\nTotal Cost of Phone Services (R$^2$={})'.format(str(coef2)))

    coef3 = r2(data['nightlights'], data['cluster_hh_f34'])
    g = sns.regplot(x="nightlights", y="cluster_hh_f34", data=data, ax=ax3)
    g.set(xlabel='Luminosity (DN)', ylabel='Number of Cell Phones',
        title='Luminosity vs Total Cell \nPhones per HH (R$^2$={})'.format(str(coef3)))

    data = data.dropna(subset=['nightlights', 'cluster_hh_f35'])
    coef4 = r2(data['nightlights'].dropna(), data['cluster_hh_f35'].dropna())
    g = sns.regplot(x="nightlights", y="cluster_hh_f35", data=data, ax=ax4)
    g.set(xlabel='Luminosity (DN)', ylabel='Annual consumption ($)',
        title='Luminosity vs Annual Consumption\nof Phone and Fax Services (R$^2$={})'.format(str(coef4)))

    fig.tight_layout()
    fig.savefig(os.path.join(BASE_PATH, '..', 'vis','figures', 'regplot.png'))

    return print('Completed regplot')


def results(data):

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(
        4, 2, figsize=(10,10))

    g = sns.regplot(x="cons_pred_10km", y="cons", data=data, ax=ax1)
    g.set(xlabel='Predicted Consumption ($)', ylabel='Consumption ($)',
        title='Consumption vs \n Predicted Consumption (Nightlights)')

    g = sns.regplot(x="predicted_cons", y="cons", data=data, ax=ax2)
    g.set(xlabel='Predicted Consumption ($ per month)',
        ylabel='Consumption ($ per month)',
        title='Consumption vs \n Luminosity (CNN)')

    g = sns.regplot(x="predicted_phone_cons", y="cluster_phone_cons",
        data=data, ax=ax4)
    g.set(xlabel='Annual Spending ($)',
        ylabel='Predicted Annual Spending ($)',
        title='Predicted Spending on Phone Services (rexp_cat083) (CNN)')

    g = sns.regplot(x="predicted_cluster_hh_f35", y="cluster_hh_f35",
        data=data, ax=ax6)
    g.set(xlabel='Annual Cost ($)',
        ylabel='Annual Predicted Cost ($)',
        title='Predicted Annual Cost of Cell Phone Services (hh_f35) (CNN)')

    g = sns.regplot(x="predicted_cluster_hh_f34", y="cluster_hh_f34",
        data=data, ax=ax8)
    g.set(xlabel='Cell Phones per Household',
        ylabel='Predicted Cell Phones per Household',
        title='Predicted Cell Phones (hh_f34) (CNN)')

    fig.tight_layout()
    fig.savefig(os.path.join(BASE_PATH, '..', 'vis','figures', 'regplot.png'))

    return print('Completed regplot')


if __name__ == '__main__':

    create_folders()

    data_1 = pd.read_csv(os.path.join(BASE_PATH, 'processed', 'lsms-cluster-2016.csv'))

    data_1 = data_1[pd.notnull(data_1['cons_pred_10km'])]

    data_2 =  pd.read_csv(os.path.join(BASE_PATH, 'processed', 'cluster_cnn_predictions.csv'))

    data = pd.merge(data_1, data_2, on=['lat', 'lon'])

    create_regplot(data)
