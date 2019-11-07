from env import host, user, password
from acquire import get_zillow
from linear_impute_landtax_lotsize import linear_impute
import pandas as pd
import numpy as np

# from bayes methodologies repo, fills zeros in nulls
def fill_zero(df, cols):
    df.loc[:, cols] = df.loc[:, cols].fillna(value=0)
    return df

# from bayes methodologies repo, fills zeros in nulls
def handle_missing_values(df, prop_required_column = .5, prop_required_row = .75):
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

# drop ID and parcelid columns
def drop_id(df):
    df = df.drop(columns=['parcelid', 'id'])
    return df

# drop rows in which these columns are 0 or nan
# use for bedcnt, bathcnt, calculatedfinishedsquarefeet
# drop bed = 0
def drop_bad_zeros(df, cols):
    fill_zero(df, cols)
    for col in cols:
        df = df.drop(df[df[col] == 0].index)
    return df

# OUTLIERS TO DROP
# baths > 7, beds > 7, sqft > 10k, garages > 5, sqft > 10k, lot > 30000, strucvalue = 0, strucvalue > 1e6
# Hold off on dropping beds/baths/garages until after we drop the other outliers
# drop_dict must be formatted as follows: {'var_name': {'under': value, 'over': value}}
# Under is inclusive, over is exclusive.
def drop_outliers(df, drop_dict):
    for var in drop_dict:
        for key in var:
            if key == 'over':
                df = df.drop((df[df[var] > drop_dict[var][key]]).index)
            elif key == 'under':
                df = df.drop((df[df[var] <= drop_dict[var][key]]).index)
    return df

def prep_zillow(df, outliers=True):
    prep = drop_id(df)

    # drop columns that have less than 10% non-null value, then rows that have less than 60% non-null values
    prep = handle_missing_values(prep, .1, .6)

    # drop columns that appear to provide little information
    prep.drop(columns=['assessmentyear', 'unitcnt', 'finishedsquarefeet12', 'propertylandusetypeid', 'rawcensustractandblock', 'censustractandblock',
                        'threequarterbathnbr', 'pooltypeid7'], inplace=True)

    # drop rows that have 0/null beds, baths, or sqft
    prep = drop_bad_zeros(prep, ['bedroomcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet'])

    # dictionary for shorter names
    lazy = {'logerror': 'logerror', 'transactiondate': 'date', 'airconditioningtypeid': 'ac', 'bathroomcnt': 'baths', 'bedroomcnt': 'beds',
            'buildingqualitytypeid': 'quality', 'calculatedbathnbr': 'calculatedbathnbr', 'calculatedfinishedsquarefeet': 'sqft',
            'fips': 'fips', 'fireplacecnt': 'fireplace', 'fullbathcnt': 'fullbaths', 'garagecarcnt': 'garage', 'garagetotalsqft': 'garagesqft',
            'heatingorsystemtypeid': 'heating', 'latitude': 'lat', 'longitude': 'long', 'lotsizesquarefeet': 'lotsqft', 'poolcnt': 'pool', 'pooltypeid7': 'pooltype',
            'propertycountylandusecode': 'usecode', 'propertyzoningdesc': 'zoning', 'regionidcity': 'city', 'regionidcounty': 'altcounty',
            'regionidneighborhood': 'neighborhood', 'regionidzip': 'zip', 'roomcnt': 'rooms', 'yearbuilt':'year',
            'numberofstories': 'stories', 'structuretaxvaluedollarcnt': 'strucvalue', 'taxvaluedollarcnt': 'value', 'landtaxvaluedollarcnt': 'landvalue',
            'taxamount': 'tax'}
    prep = prep.rename(columns=lazy)

    # Drop outliers
    # Outliers were determined by looking at value_counts and distribution vizzes
    if outliers:
        drop_dict = {'sqft': {'over': 1e4},
                    'lotsqft': {'over': 3e4},
                    'strucvalue': {'over': 1e6, 'under': 0}, 
                    'beds': {'over': 7}, 
                    'baths': {'over': 7}, 
                    'garage': {'over': 5}}
        prep = drop_outliers(prep, drop_dict)

    # IMPUTATIONS
    # We noticed there was a roughly linear correlation between land value and lot size
    prep = linear_impute(prep, 'landvalue', 'lotsqft')

    # We believe nulls here represent no's or zeros
    to_zero = ['fireplace', 'fullbaths', 'pool']
    prep = fill_zero(prep, to_zero)
    return prep

if __name__ == '__main__':
    # zillow = get_zillow()
    zillow = pd.read_csv('zillow.csv')
    prep = prep_zillow(zillow)