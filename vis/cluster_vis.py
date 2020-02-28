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
import geoio
import math

CONFIG_DATA = configparser.ConfigParser()
CONFIG_DATA.read(os.path.join(os.path.dirname(__file__), '..', 'scripts','script_config.ini'))
BASE_PATH = CONFIG_DATA['file_locations']['base_path']

CONFIG_COUNTRY = configparser.ConfigParser()
CONFIG_COUNTRY.read('script_config.ini')
COUNTRY = CONFIG_COUNTRY['DEFAULT']['COUNTRY']
SHAPEFILE_DIR = f'countries/{COUNTRY}/shapefile'
GRID_DIR = f'countries/{COUNTRY}/grid'
RESULTS_DIR = f'countries/{COUNTRY}/results'
LSMS_DIR = f'data/LSMS/{COUNTRY}/input'
NIGHTLIGHTS_DIR = 'data/Nightlights/2013'
WORLDPOP = 'data/world_population'
CLUSTER_DATA_DIR = f'data/LSMS/{COUNTRY}/processed/cluster_data.csv'
CLUSTER_PREDICTIONS_DIR = f'data/LSMS/{COUNTRY}/output/cluster_predictions.csv'

# Purchasing Power Adjustment
PPP = float(CONFIG_COUNTRY['DEFAULT']['PPP'])

CNN_LSMS_CLUSTER_FEATS = f'data/LSMS/{COUNTRY}/processed/cluster_feats.npy'
CNN_LSMS_CLUSTER_ORDER = f'data/LSMS/{COUNTRY}/processed/cluster_order.pkl'


def create_folders():

    path = os.path.join(BASE_PATH, '..', 'vis','figures')
    if not os.path.exists(path):
        os.mkdir(path)


def prepare_data():
    """
    Preprocessing script.

    """
    # read aggregared consumption data
    df = pd.read_stata(os.path.join(LSMS_DIR, 'IHS4 Consumption Aggregate.dta'))#[:10]
    df['persons_in_household'] = df['rexpagg'] / df['rexpaggpc']
    df['annual_consumption_hh'] = df['rexpagg']

    # account for purchasing power parity
    df['annual_consumption_hh'] /= PPP
    df['annual_phone_consumption_hh'] = df['rexp_cat083']
    df['annual_phone_consumption_hh'] /= PPP

    # subset consumption columns
    df = df[[
        'case_id', 'annual_consumption_hh',
        'annual_phone_consumption_hh', 'persons_in_household'
        ]]

    # load in data with geoinformation
    path = os.path.join('HouseholdGeovariables_stata11', 'HouseholdGeovariablesIHS4.dta')
    df_geo = pd.read_stata(os.path.join(LSMS_DIR, path))#[:100]

    # subset household id with coordinates
    df_cords = df_geo[['case_id', 'HHID', 'lat_modified', 'lon_modified']]
    df_cords.rename(columns={'lat_modified': 'lat', 'lon_modified': 'lon'}, inplace=True)

    # load in specific cellphone questions
    df_hhf = pd.read_stata(os.path.join(LSMS_DIR, 'HH_MOD_F.dta'))
    df_hhf = df_hhf[['case_id', 'HHID', 'hh_f34', 'hh_f35']]

    df_hhf.rename(
        columns={'hh_f34': 'cellphones_ph',
        'hh_f35': 'estimated_annual_phone_cost_ph'},
        inplace=True)

    # add coordinates to data without geoinfo
    df = pd.merge(df, df_cords[['case_id', 'HHID']], on='case_id')
    df_combined = pd.merge(df, df_cords, on=['case_id', 'HHID'])
    df_combined = pd.merge(df_combined, df_hhf, on=['case_id', 'HHID'])

    warning = 'Have a household with null people'
    assert df_combined['persons_in_household'].isna().sum() == 0, print(warning)

    df_clusters = df_combined.groupby(['lat', 'lon']).sum().reset_index()#[:100]

    # Data is per household
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
    for _, r in df_clusters.iterrows():

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

    filename = 'ppp_2020_1km_Aggregated.tif'
    img = geoio.GeoImage(os.path.join(WORLDPOP, filename))
    im_array = np.squeeze(img.get_data())

    cluster_population = []
    for _, r in df_clusters.iterrows():

        min_lat, min_lon, max_lat, max_lon = create_space(r.cluster_lat, r.cluster_lon)
        xminPixel, yminPixel = img.proj_to_raster(min_lon, min_lat)
        xmaxPixel, ymaxPixel = img.proj_to_raster(max_lon, max_lat)

        xminPixel, xmaxPixel = min(xminPixel, xmaxPixel), max(xminPixel, xmaxPixel)
        yminPixel, ymaxPixel = min(yminPixel, ymaxPixel), max(yminPixel, ymaxPixel)

        xminPixel, yminPixel, xmaxPixel, ymaxPixel = (
            int(xminPixel), int(yminPixel), int(xmaxPixel), int(ymaxPixel)
        )
        cluster_population.append(
            round(im_array[yminPixel:ymaxPixel,xminPixel:xmaxPixel].mean()))

    df_clusters['cluster_population_density_1km2'] = cluster_population

    return df_clusters


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


def r2(x, y):

    coef = round(np.corrcoef(x, y)[0, 1]**2, 3)

    return coef


def create_regplot(data):

    data = data[data.cluster_annual_consumption_pc <= 10000]

    data['cluster_monthly_consumption_pc'] = data['cluster_annual_consumption_pc'] / 12
    data['cluster_monthly_phone_consumption_pc'] = data['cluster_annual_phone_consumption_pc'] / 12
    data['cluster_monthly_phone_cost_pc'] = data['cluster_estimated_annual_phone_cost_pc'] / 12

    bins = [0, 400, 800, 1200, 4020]
    labels = [5, 15, 25, 60]
    data['pop_density_binned'] = pd.cut(data['cluster_population_density_1km2'], bins=bins, labels=labels)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,8))

    coef1 = r2(data['cluster_nightlights'], data['cluster_monthly_consumption_pc'])
    g = sns.regplot(y="cluster_nightlights", x="cluster_monthly_consumption_pc",
        data=data, ax=ax1, scatter_kws={'alpha': 0.2, 'color':'blue'},
        line_kws={'alpha': 0.5, 'color':'black'})
    g.set(ylabel='Luminosity (DN)', xlabel='Consumption per capita ($ per month)',
        title='Luminosity vs\nConsumption (R$^2$={})'.format(str(coef1)),
        ylim=(0, 50))

    coef2 = r2(data['cluster_nightlights'], data['cluster_monthly_phone_consumption_pc'])
    g = sns.regplot(y="cluster_nightlights", x="cluster_monthly_phone_consumption_pc",
        data=data, ax=ax2, scatter_kws={'alpha': 0.2, 'color':'blue'},
        line_kws={'alpha': 0.5, 'color':'black'})
    g.set(ylabel='Luminosity (DN)', xlabel='Total Cost ($ per month)',
        title='Luminosity vs\nTotal Cost of Phone Services (R$^2$={})'.format(str(coef2)),
        ylim=(0, 50))

    #'hh_f34': 'cellphones_pc',
    coef3 = r2(data['cluster_nightlights'], data['cluster_cellphones_pc'])
    g = sns.regplot(y="cluster_nightlights", x="cluster_cellphones_pc",
        data=data, ax=ax3, scatter_kws={'alpha': 0.2, 'color':'blue'},
        line_kws={'alpha': 0.5, 'color':'black'})
    g.set(ylabel='Luminosity (DN)', xlabel='Number of Cell Phones',
        title='Luminosity vs Total Cell \nPhones per HH (R$^2$={})'.format(str(coef3)),
        ylim=(0, 50))

    #'hh_f35': 'estimated_annual_phone_cost_pc'
    data = data.dropna(subset=['cluster_nightlights', 'cluster_monthly_phone_cost_pc'])
    coef4 = r2(data['cluster_nightlights'].dropna(), data['cluster_monthly_phone_cost_pc'].dropna())
    g = sns.regplot(y="cluster_nightlights", x="cluster_monthly_phone_cost_pc",
        data=data, ax=ax4, scatter_kws={'alpha': 0.2, 'color':'blue'},
        line_kws={'alpha': 0.5, 'color':'black'})
    g.set(ylabel='Luminosity (DN)', xlabel='Annual consumption ($)',
        title='Luminosity vs Annual Consumption\nof Phone and Fax Services (R$^2$={})'.format(str(coef4)),
        ylim=(0, 50))

    fig.tight_layout()
    fig.savefig(os.path.join(BASE_PATH, '..', 'vis','figures', 'regplot.png'))

    return print('Completed regplot')


def results(data):

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(
        4, 2, figsize=(10,10))

    g = sns.regplot(x="cons_pred_10km", y="cons", data=data, ax=ax1)
    g.set(xlabel='Predicted Consumption ($)', ylabel='Consumption ($)',
        title='Consumption vs \n Predicted Consumption (cluster_nightlights)')

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

    data = prepare_data()

    create_regplot(data)
