
RANDOM_SEED = 7 # constant for reproducability

# for training
TRAINING_CONFIG = {
    'TYPE': 'single_country', # single_country or country_held_out
    'COUNTRY': 'malawi_2016', # malawi_2016, ethiopia_2015
    'METRIC': 'house_has_cellphone' # house_has_cellphone or est_monthly_phone_cost_pc

}

# for prediction maps
PREDICTION_MAPS_CONFIG = {
    'COUNTRY_ABBRV': 'MWI', # MWI or ETH, but could be any country code in the world

    # what type of model to use to predict this country
    'TYPE': 'single_country',
    'COUNTRY': 'malawi_2016',
    'METRIC': 'house_has_cellphone'
}
