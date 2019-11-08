import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, LabelEncoder, OneHotEncoder, QuantileTransformer, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor, LassoCV, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from acquire import get_zillow
from prepare import prep_zillow
from split_scale import scale

# BASELINE
df = get_zillow()
df = prep_zillow(df)
df = df[['logerror','sqft','fips','lat', 'long', 'lotsqft', 'age', 'tax']]
train1, test1 = train_test_split(df, random_state = 123)

X1_train = train1[['sqft', 'lotsqft', 'tax', 'age']]
y1_train = train1[['logerror']]

lm1 = LinearRegression()

lm1.fit(X1_train, y1_train)

y_pred_lm1 = lm1.predict(X1_train)


# Evaluate
mean_squared_error(y1_train, y_pred_lm1)**1/2

r2_score(y1_train, y_pred_lm1)

y1_train.index.values.reshape(-1,1)

y_pred_baseline = np.array([y1_train.mean()[0]]*len(y1_train))

# Model experimentation

zillow = get_zillow()
zillow = prep_zillow(zillow)

# ENCODE FIPS AND CITY
# actually not sure how to do that with city... there are many different values. might have to map them or something?
# 6037 is LA, 6059 is Orange, 6111 is Ventura
zillow['la'] = int(zillow['fips'] == 6037)
zillow['orange'] = int(zillow['fips'] == 6059)
zillow['ventura'] = int(zillow['fips'] == 6111)

# split data
train, test = train_test_split(zillow, test_size=.30, random_state=123)

features = ['baths', 'beds', 'sqft', 'fireplace', 'lat', 'long', 'lotsqft', 'pool',
            'tax', 'age', 'strucvaluebysqft', 'landvaluebysqft']
target = 'logerror'

# SCALE
uniform = ['baths', 'beds', 'sqft', 'lotsqft', 'strucvaluebysqft', 'landvaluebysqft', 'tax']
minmax = ['lat', 'long', 'age']
uniform_scaler, train, test = scale(train, test, uniform, 'uniform')
minmax_scaler, train, test = scale(train, test, minmax, scaler='minmax')

# CLUSTERS

# MODEL