
RANDOM_SEED = 7 # constant for reproducability

# for training
TRAINING_CONFIG = {
    'TYPE': 'single_country', # single_country or country_held_out
    'COUNTRY': 'malawi_2016', # malawi_2016, ethiopia_2015
    'METRIC': 'est_monthly_phone_cost_pc' # house_has_cellphone or est_monthly_phone_cost_pc
}

# for prediction maps
VIS_CONFIG = {
    'COUNTRY_NAME': "Ethiopia", # malawi_2016 -> Malawi, ethiopia_2015 -> Ethiopia
    'COUNTRY_ABBRV': 'MWI', # MWI or ETH, but could be any country code in the world

    # what type of model to use to predict this country
    'TYPE': 'single_country',
    'COUNTRY': 'ethiopia_2015',
    'METRIC': 'est_monthly_phone_cost_pc'
}
