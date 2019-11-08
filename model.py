import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, LabelEncoder, OneHotEncoder, QuantileTransformer, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor, LassoCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures 
from acquire import get_zillow
from prepare import prep_zillow
from split_scale import scale


zillow = get_zillow()
zillow = prep_zillow(zillow)

# ENCODE FIPS AND CITY
# actually not sure how to do that with city... there are many different values. might have to map them or something?
# 6037 is LA, 6059 is Orange, 6111 is Ventura
zillow['la'] = int(zillow['fips'] == 6037)
zillow['orange'] = int(zillow['fips'] == 6059)
zillow['ventura'] = int(zillow['fips'] == 6111)

# split data
train, test = train_test_split(zillow, test_size=.30)

features = ['baths', 'beds', 'sqft', 'fireplace', 'lat', 'long', 'lotsqft', 'pool', 'strucvalue', 'value', 'landvalue', 'tax', 'age']
target = 'logerror'

# SCALE
uniform = ['baths', 'beds', 'sqft', 'lotsqft', 'strucvalue', 'value', 'landvalue', 'tax']
minmax = ['lat', 'long', 'age']
uniform_scaler, train, test = scale(train, test, uniform, 'uniform')
minmax_scaler, train, test = scale(train, test, minmax, scaler='minmax')

# CLUSTERS