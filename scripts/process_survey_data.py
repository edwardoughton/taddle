'''
Parse the metrics using the survey per cluster

Dataframes saved to data/<country>/processed/clusters.csv
Should have the following columns:
- country
- cluster_lat
- cluster_lon
- house_has_cellphone
- est_monthly_phone_cost_pc
'''


import pandas as pd
import numpy as np
import os

BASE_DIR = '.'
# repo imports
import sys
sys.path.append('.')
from utils import create_space

COUNTRIES_DIR = os.path.join(BASE_DIR, 'data', 'countries')

'''
The goal of each of these functions is to output a dataframe with the following columns:
country, cluster_lat, cluster_lon, cons_pc

Each row should represent one cluster by combining the household data
'''

def process_malawi():
    lsms_dir = os.path.join(COUNTRIES_DIR, 'malawi_2016', 'LSMS')
    consumption_file = 'IHS4 Consumption Aggregate.csv'
    hhsize_col = 'hhsize' # people in household

    geovariables_file = 'HouseholdGeovariables_csv/HouseholdGeovariablesIHS4.csv'
    lat_col = 'lat_modified'
    lon_col = 'lon_modified'

    # purchasing power parity for malawi in 2016 (https://data.worldbank.org/indicator/PA.NUS.PRVT.PP?locations=MW)
    ppp = 215.182
    
    for file in [consumption_file, geovariables_file]:
        assert os.path.isfile(os.path.join(lsms_dir, file)), print(f'Could not find {file}')
    
    # persons per household
    df_pph = pd.read_csv(os.path.join(lsms_dir, consumption_file))
    df_pph['pph'] = df_pph[hhsize_col]
    df_pph = df_pph[['case_id', 'pph']]

    df_phone_data = pd.read_csv(os.path.join(lsms_dir, 'hh_mod_f.csv'))
    df_phone_data['house_has_cellphone'] = df_phone_data['hh_f34'] >= 1
    df_phone_data['est_monthly_phone_cost_ph'] = df_phone_data['hh_f35'] / ppp / 12 # monthly, in dollars
    df_phone_data = df_phone_data[['case_id', 'house_has_cellphone', 'est_monthly_phone_cost_ph']]

    df_geo = pd.read_csv(os.path.join(lsms_dir, geovariables_file))
    df_cords = df_geo[['case_id', lat_col, lon_col]]
    df_cords.rename(columns={lat_col: 'cluster_lat', lon_col: 'cluster_lon'}, inplace=True)
    df_combined = pd.merge(df_pph, df_cords, on='case_id')
    df_combined = pd.merge(df_combined, df_phone_data, on='case_id')
    df_combined.drop(['case_id'], axis=1, inplace=True)
    df_combined['est_monthly_phone_cost_ph'].fillna(0, inplace=True) # if its na, that means the house has no phone
    
    cluster_lats = []
    cluster_lons = []
    house_has_cellphone = []
    est_monthly_phone_cost_pc = []
    groups = df_combined.groupby(['cluster_lat', 'cluster_lon'])
    for (cluster_lat, cluster_lon), group in groups:
        cluster_lats.append(cluster_lat)
        cluster_lons.append(cluster_lon)
        house_has_cellphone.append(group['house_has_cellphone'].mean())
        est_monthly_phone_cost_pc.append(group['est_monthly_phone_cost_ph'].sum() / len(group))

    df_clusters = pd.DataFrame.from_dict({'cluster_lat': cluster_lats, 'cluster_lon': cluster_lons, 'house_has_cellphone': house_has_cellphone, 'est_monthly_phone_cost_pc': est_monthly_phone_cost_pc})
    df_clusters['country'] = 'malawi_2016'
    return df_clusters

def process_ethiopia():
    lsms_dir = os.path.join(COUNTRIES_DIR, 'ethiopia_2015', 'LSMS')
    consumption_file = 'Consumption Aggregate/cons_agg_w3.csv'
    hhsize_col = 'hh_size' # people in household

    geovariables_file = 'Geovariables/ETH_HouseholdGeovars_y3.csv'
    lat_col = 'lat_dd_mod'
    lon_col = 'lon_dd_mod'

    # purchasing power parity for ethiopia in 2015 (https://data.worldbank.org/indicator/PA.NUS.PRVT.PP?locations=ET)
    ppp = 7.882
    
    for file in [consumption_file, geovariables_file]:
        assert os.path.isfile(os.path.join(lsms_dir, file)), print(f'Could not find {file}')
    
    # persons per household
    df_pph = pd.read_csv(os.path.join(lsms_dir, consumption_file))
    df_pph['pph'] = df_pph[hhsize_col]
    df_pph = df_pph[['household_id2', 'pph']]

    df_phone_data = pd.read_csv(os.path.join(lsms_dir, 'Household/sect9_hh_w3.csv'))
    df_phone_data['house_has_cellphone'] = df_phone_data['hh_s9q22'] == 1 # 1 means Yes, 2 means No
    df_phone_data['est_monthly_phone_cost_ph'] = df_phone_data['hh_s9q23'] / ppp / 12 # monthly, in dollars
    df_phone_data = df_phone_data[['household_id2', 'house_has_cellphone', 'est_monthly_phone_cost_ph']]

    df_geo = pd.read_csv(os.path.join(lsms_dir, geovariables_file))
    df_cords = df_geo[['household_id2', lat_col, lon_col]]
    df_cords.rename(columns={lat_col: 'cluster_lat', lon_col: 'cluster_lon'}, inplace=True)
    df_combined = pd.merge(df_pph, df_cords, on='household_id2')
    df_combined = pd.merge(df_combined, df_phone_data, on='household_id2')
    df_combined.drop(['household_id2'], axis=1, inplace=True)
    # if no phones, no cost; this will replace some nulls
    df_combined['est_monthly_phone_cost_ph'].loc[df_combined['house_has_cellphone'] == False] = 0
    df_combined.dropna(inplace=True) # we can't use any remaining null values
    
    cluster_lats = []
    cluster_lons = []
    house_has_cellphone = []
    est_monthly_phone_cost_pc = []
    groups = df_combined.groupby(['cluster_lat', 'cluster_lon'])
    for (cluster_lat, cluster_lon), group in groups:
        cluster_lats.append(cluster_lat)
        cluster_lons.append(cluster_lon)
        house_has_cellphone.append(group['house_has_cellphone'].mean())
        est_monthly_phone_cost_pc.append(group['est_monthly_phone_cost_ph'].sum() / len(group))

    df_clusters = pd.DataFrame.from_dict({'cluster_lat': cluster_lats, 'cluster_lon': cluster_lons, 'house_has_cellphone': house_has_cellphone, 'est_monthly_phone_cost_pc': est_monthly_phone_cost_pc})
    df_clusters['country'] = 'ethiopia_2015'
    return df_clusters


def save_df(df, country):
    savepath = os.path.join(COUNTRIES_DIR, country, 'processed', 'clusters.csv')
    os.makedirs(os.path.join(COUNTRIES_DIR, country, 'processed'), exist_ok=True)
    print(f"saving to {savepath}")
    df.to_csv(savepath, index=False)


if __name__ == '__main__':
    if not os.path.exists(os.path.join(COUNTRIES_DIR, 'malawi_2016', 'processed', 'clusters.csv')):
        print('processing malawi_2016')
        df_mw = process_malawi()
        save_df(df_mw, 'malawi_2016')
    else:
        print('malawi_2016 is already processed')

    if not os.path.exists(os.path.join(COUNTRIES_DIR, 'ethiopia_2015', 'processed', 'clusters.csv')):
        print('processing ethiopia_2015')
        df_eth = process_ethiopia()
        save_df(df_eth, 'ethiopia_2015')
    else:
        print('ethiopia_2015 is already processed')
        
        