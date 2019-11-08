import pandas as pd
import numpy as np
from acquire import get_zillow
from prepare import prep_zillow
from sklearn.preprocessing import PowerTransformer, LabelEncoder, OneHotEncoder, QuantileTransformer, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor, LassoCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures 


zillow = get_zillow()
zillow = prep_zillow(zillow)

# split data
train, test = train_test_split(zillow, test_size=.30)

features = ['baths', 'beds', 'sqft', 'fireplace', 'lat', 'long', 'lotsqft', 'pool', 'strucvalue', 'value', 'landvalue', 'tax', 'age']
target = 'logerror'

# SCALE

# ENCODE FIPS AND CITY
# actually not sure how to do that with city... there are many different values. might have to map them or something?