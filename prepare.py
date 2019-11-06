from env import host, user, password
from acquire import get_zillow
import pandas as pd
import numpy as np

# from bayes methodologies repo, fills zeros in nulls
def fill_zero(df, cols):
    df.fillna(value=0, inplace=True)
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
    df.drop(columns=['parcelid', 'id'], inplace=True)
    return df

# drop rows in which these columns are 0 or nan
# use for bedcnt, bathcnt, calculatedfinishedsquarefeet
# drop bed = 0
def drop_bad_zeros(df, cols):
    fill_zero(df, cols)
    for col in cols:
        df = df.drop((df[col] == 0).index)
    return df

# FEATURES IDEAS:
# tax rate, house age

def prep_zillow(df):
    prep = drop_id(df)
    prep = handle_missing_values(prep, .1, .6)
    # drop columns that appear to provide little information
    prep.drop(columns=['assessmentyear', 'unitcnt', 'finishedsquarefeet12', 'propertylandusetypeid', 'rawcensustractandblock', ], inplace=True)
    prep = drop_bad_zeros(prep, ['bedroomcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet'])
    to_zero = []
    prep = fill_zero(prep, to_zero)
    lazy = {}
    prep = prep.rename(lazy)
    return prep

if __name__ == '__main__':
    zillow = get_zillow()
    prep = prep_zillow(zillow)