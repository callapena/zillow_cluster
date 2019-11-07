from env import host, user, password
import pandas as pd
import numpy as np
from acquire import get_zillow
from prepare import prep_zillow

# YES/NO AIRCONDITIONING, FIREPLACE, HEATING

# COMBINED BATHS & BEDS

# CONVERT CATEGORICAL NUMBERS TO CATEGORIES OR STRINGS
# ONEHOTENCODER

def feature_prep(df):
    # Convert year built to house age
    df['age'] = 2017 - df['year']
    df = df.drop(columns='year')

    # Make tax rate variable in place of tax
    df['tax'] = df['tax'] / df['value']

    # Make fireplace boolean-ish
    df['fireplace'] = (df[['fireplace']] > 0).astype(int)
    return df

def scale(df):
    return df