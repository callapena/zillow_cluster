from env import host, user, password
from acquire import get_zillow
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
# baths > 7, beds > 7, sqft > 10k, garages > 5, sqft > 10k, lot > 30000

def prep_zillow(df, drop_outliers=True):
    prep = drop_id(df)
    prep = handle_missing_values(prep, .1, .6)
    # drop columns that appear to provide little information
    prep.drop(columns=['assessmentyear', 'unitcnt', 'finishedsquarefeet12', 'propertylandusetypeid', 'rawcensustractandblock', 'censustractandblock',
                        'threequarterbathnbr'], inplace=True)
    prep = drop_bad_zeros(prep, ['bedroomcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet'])
    lazy = {'logerror': 'logerror', 'transactiondate': 'date', 'airconditioningtypeid': 'ac', 'bathroomcnt': 'baths', 'bedroomcnt': 'beds',
            'buildingqualitytypeid': 'quality', 'calculatedbathnbr': 'calculatedbathnbr', 'calculatedfinishedsquarefeet': 'sqft',
            'fips': 'fips', 'fireplacecnt': 'fireplace', 'fullbathcnt': 'fullbaths', 'garagecarcnt': 'garage', 'garagetotalsqft': 'garagesqft',
            'heatingorsystemtypeid': 'heating', 'latitude': 'lat', 'longitude': 'long', 'lotsizesquarefeet': 'lotsqft', 'poolcnt': 'pool', 'pooltypeid7': 'pooltype',
            'propertycountylandusecode': 'usecode', 'propertyzoningdesc': 'zoning', 'regionidcity': 'city', 'regionidcounty': 'altcounty',
            'regionidneighborhood': 'neighborhood', 'regionidzip': 'zip', 'roomcnt': 'rooms', 'yearbuilt':'year',
            'numberofstories': 'stories', 'structuretaxvaluedollarcnt': 'strucvalue', 'taxvaluedollarcnt': 'value', 'landtaxvaluedollarcnt': 'landvalue',
            'taxamount': 'tax'}
    prep = prep.rename(columns=lazy)
    to_zero = ['fireplace', 'fullbaths']
    prep = fill_zero(prep, to_zero)
    return prep

if __name__ == '__main__':
    zillow = get_zillow()
    prep = prep_zillow(zillow)