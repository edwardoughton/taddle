"""
Build a model that uses hand-picked features that we think an image could detect.
This creates a "gold standard" for a 3-band RGB image-based model.

Written by Jatin Mathur.

Winter 2020

"""

import configparser
import os
import pandas as pd
import numpy as np

# repo imports
import sys
sys.path.append('.')
from utils import merge_on_lat_lon
from scripts.create_ridge_models import predict_metric

CONFIG = configparser.ConfigParser()
CONFIG.read('script_config.ini')

COUNTRY = CONFIG['DEFAULT']['COUNTRY']

LSMS_DIR = f'data/LSMS/{COUNTRY}/input'
CLUSTER_DATA_DIR = f'data/LSMS/{COUNTRY}/processed/cluster_data.csv'

# Purchasing Power Adjustment
PPP = float(CONFIG['DEFAULT']['PPP'])
SEED = int(CONFIG['DEFAULT']['SEED'])

np.random.seed(SEED)

# here are all the features I have hand-picked

# rooms = df_hhf['hh_f10']
# roof = df_hhf['hh_f08']

# # all distance infrasturcture metrics
# road_type = df_com['com_cd01']
# dist_daily_market = df_com['com_cd16']
# dist_larger_weekly = df_com['com_cd18a']
# dist_perm_admarc = df_com['com_cd20a']
# dist_post_office = df_com['com_cd22a']
# dist_telephone = df_com['com_cd24a']
# dist_gov_prim_school = df_com['com_cd27a']
# dist_gov_sec_school = df_com['com_cd36a']
# dist_comm_sec_school = df_com['com_cd40a']
# dist_medicines = df_com['com_cd49a']
# dist_health_clinic = df_com['com_cd51a']
# dist_doctor = df_com['com_cd60a']
# dist_bank = df_com['com_cd67a']
# dist_microfinance = df_com['com_cd69a']

# dist_agric_exten_officer = df_com2['com_cf08a']

# dist_admarc_outlet = df_geo['dist_admarc']
# dist_agric_market = df_geo['dist_agmrkt']
# dist_tobacco_auction = df_geo['dist_auction']
# dist_boma = df_geo['dist_boma']
# dist_border = df_geo['dist_borderpost']
# dist_popcenter = df_geo['dist_popcenter']
# dist_road = df_geo['dist_road']

# dist_hh = df_plot['dist_hh']

# # temp
# mean_temp = df_geo['af_bio_1']
# mean_temp_wet_q = df_geo['af_bio_8']

# # rain
# mean_rain = df_geo['af_bio_12']
# mean_rain_wet_month = df_geo['af_bio_13']
# mean_rain_wet_q = df_geo['af_bio_16']

def nan_handler(df):
    nas = df.isna().sum()
    for c in df:
        if nas[c] > 0:
            df[c] = df[c].fillna(df[c].median())
    return df

def prepare_data():
    """
    Preprocessing script that hand-picks all the features

    """
    assert os.path.isfile(CLUSTER_DATA_DIR), print('Make sure you have processed the cluster data')

    print('preprocessing...')
    df_geo = pd.read_stata(os.path.join(LSMS_DIR, 'HouseholdGeovariables_stata11/HouseholdGeovariablesIHS4.dta'))
    df_hhf = pd.read_stata(os.path.join(LSMS_DIR, 'HH_MOD_F.dta'))
    df_plot = pd.read_stata(os.path.join(LSMS_DIR, 'PlotGeovariablesIHS4.dta'))

    df_com = pd.read_stata(os.path.join(LSMS_DIR, 'COM_CD.dta'))
    df_com2 = pd.read_stata(os.path.join(LSMS_DIR, 'COM_CF1.dta'))

    df_tie = pd.read_stata(os.path.join(LSMS_DIR, 'IHS4 Consumption Aggregate.dta'))[['case_id', 'ea_id']]

    hhf_input = df_hhf[['case_id', 'hh_f10', 'hh_f08']]
    com_input = df_com[['ea_id', 'com_cd01', 'com_cd16', 'com_cd18a', 'com_cd20a', 'com_cd22a', 'com_cd24a',
                    'com_cd27a', 'com_cd36a', 'com_cd40a', 'com_cd49a', 'com_cd51a', 'com_cd60a', 'com_cd67a',
                    'com_cd69a']]

    com2_input = df_com2[['ea_id', 'com_cf08a']]

    geo_input = df_geo[['case_id', 'dist_admarc', 'dist_agmrkt', 'dist_auction', 'dist_boma', 'dist_borderpost',
                    'dist_popcenter', 'dist_road', 'af_bio_1', 'af_bio_8', 'af_bio_12', 'af_bio_13', 'af_bio_16', 
                    'lat_modified', 'lon_modified']]
    geo_input.rename(columns={'lat_modified': 'cluster_lat', 'lon_modified': 'cluster_lon'}, inplace=True)
    geo_input.dropna(inplace=True)

    plot_input = df_plot[['case_id', 'dist_hh']]

    df_processed = pd.read_csv(os.path.join(CLUSTER_DATA_DIR))

    df_merge = merge_on_lat_lon(df_processed, geo_input, how='left')
    df_merge = pd.merge(df_merge, hhf_input, on='case_id', how='left')
    df_merge = pd.merge(df_merge, df_tie, on='case_id', how='left')
    df_merge = pd.merge(df_merge, com_input, on='ea_id', how='left')
    df_merge = pd.merge(df_merge, com2_input, on='ea_id', how='left')
    df_merge = pd.merge(df_merge, plot_input, on='case_id', how='left')

    df_merge = df_merge.drop(['case_id', 'ea_id'], axis=1)
    df_merge = pd.get_dummies(df_merge)

    clusters = df_merge.groupby(['cluster_lat', 'cluster_lon'])
    cluster_df = clusters.mean().reset_index()
    cluster_df = nan_handler(cluster_df)

    return cluster_df

def train_gold_standard(cluster_df, metrics):
    unused = ['cluster_lat', 'cluster_lon', 'cluster_persons_surveyed', 'cluster_nightlights', 'cluster_estimated_annual_phone_cost_pc']
    x = cluster_df.drop(metrics + unused, axis=1).values
    
    for metric in metrics:
        y = cluster_df[metric].values
        print(f'Training model on {metric}...')
        _, r2, _, _ = predict_metric(x, y)
        print('average r2:', r2)
        print()

if __name__ == '__main__':
    cluster_df = prepare_data()

    metrics = ['cluster_annual_consumption_pc', 'cluster_annual_phone_consumption_pc', 'cluster_cellphones_pc']
    train_gold_standard(cluster_df, metrics)
