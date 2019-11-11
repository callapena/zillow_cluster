import pandas as pd
import numpy as np
import scipy as sp
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

# Model experimentation
zillow = get_zillow()
zillow = prep_zillow(zillow)

# ENCODE FIPS AND CITY
# actually not sure how to do that with city... there are many different values. might have to map them or something?
# 6037 is LA, 6059 is Orange, 6111 is Ventura
zillow['la'] = (zillow['fips'] == 6037).astype(int)
zillow['orange'] = (zillow['fips'] == 6059).astype(int)
zillow['ventura'] = (zillow['fips'] == 6111).astype(int)

# split data
train, test = train_test_split(zillow, test_size=.30, random_state=123)
X1_train = train[['sqft', 'lotsqft', 'tax', 'age']]
y1_train = train['logerror']
test1 = test[['sqft', 'lotsqft', 'tax', 'age', 'logerror']]


features = ['beds_and_baths', 'sqft', 'fireplace', 'lat', 'long', 'lotsqft', 'pool',
            'tax', 'age', 'strucvaluebysqft', 'landvaluebysqft', 'la', 'orange', 'ventura']
target = 'logerror'

# SCALE
uniform = ['beds_and_baths', 'sqft', 'lotsqft', 'strucvaluebysqft', 'landvaluebysqft', 'tax']
minmax = ['lat', 'long', 'age']
uniform_scaler, train, test = scale(train, test, uniform, scaler='uniform')
minmax_scaler, train, test = scale(train, test, minmax, scaler='minmax')

X_train = train[features]
X_test = test[features]
y_train = train[target]
y_test = test[target]

# CLUSTERS
neighborhood = ['lat', 'long', 'strucvaluebysqft', 'landvaluebysqft']

amenities = ['beds_and_baths', 'sqft', 'lotsqft', 'age']

def cluster_exam(max_k, X_train, features):
    ks = range(1, max_k + 1)
    sse = []
    for k in ks:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X_train[features])

        # inertia: Sum of squared distanes of samplesto their closest cluster
        sse.append(kmeans.inertia_)
    print(pd.DataFrame(dict(k=ks, sse=sse)))

# cluster_exam(10, X_train, neighborhood) # Let's go with 7 or 8
# cluster_exam(10, X_train, amenities) # Probably 3, 5, or 8

X3_train = X_train.copy(deep=True)
test3 = test.copy(deep=True)

neighborhood_kmeans = KMeans(n_clusters=8)
neighborhood_kmeans.fit(X_train[neighborhood])
X3_train['neighborhood'] = neighborhood_kmeans.predict(X3_train[neighborhood])
test3['neighborhood'] = neighborhood_kmeans.predict(test3[neighborhood])

amenities_kmeans = KMeans(n_clusters=8)
amenities_kmeans.fit(X_train[amenities])
X3_train['amenities'] = amenities_kmeans.predict(X3_train[amenities])
test3['amenities'] = amenities_kmeans.predict(test3[amenities])

neighbor_feats = []
amenities_feats = []

for i in range(1, 9):
    X3_train['n' + str(i)] = (X3_train.neighborhood == i).astype(int)
    X3_train['a' + str(i)] = (X3_train.amenities == i).astype(int)
    test3['n' + str(i)] = (test3.neighborhood == i).astype(int)
    test3['a' + str(i)] = (test3.amenities == i).astype(int)
    neighbor_feats.append('n' + str(i))
    amenities_feats.append('a' + str(i))

X3_train = X3_train.drop(columns=(amenities + neighborhood + ['amenities', 'neighborhood']))
test3 = test3.drop(columns=(amenities + neighborhood + ['amenities', 'neighborhood']))

# DATA VALIDATION
bool_features = ['fireplace', 'pool', 'la', 'orange', 'ventura']

def validation(val_train, val_y_train, val_test, cols):
    to_drop = []
    val_train['logerror'] = val_y_train

    for feature in cols:
        pval = sp.stats.ttest_ind(
            val_train[val_train[feature] == 0].logerror.dropna(),
            val_train[val_train[feature] == 1].logerror.dropna())[1]

        if pval > .05:
            to_drop.append(feature)

    res_train = val_train.drop(columns=to_drop)
    res_test = val_test.drop(columns=to_drop)
    return res_train, res_test

X2_train, test2 = validation(X_train, y_train, test, bool_features)
X2_train = X2_train.drop(columns='logerror')

X4_train, test4 = validation(X3_train, y_train, test3, bool_features + neighbor_feats + amenities_feats)
X4_train = X4_train.drop(columns='logerror')

# MODEL

def evaluate(model, x_train, y_train):
    y_pred = model.predict(x_train)
    rmse = mean_squared_error(y_train, y_pred)**1/2
    return rmse

# Takes a dictionary of models and returns the best performing model
# Dictionary format example:
# model_dict = {'model_name': {'obj': model_object, 'data': data_dict}}
def best_model(model_dict):
    best = [None, 1]
    for model in model_dict:
        model_obj = model_dict[model]['obj']
        x = model_dict[model]['data']['x']
        y = model_dict[model]['data']['y']
        error = evaluate(model_obj, x, y)
        if error < best[1]:
            best = [model_dict[model], error]
    return best

def best_test(model_dict):
    best = [None, 1]
    for model in model_dict:
        model_obj = model_dict[model]['obj']
        x = model_dict[model]['data']['test'][model_dict[model]['data']['x'].columns]
        y = model_dict[model]['data']['test']['logerror']
        error = evaluate(model_obj, x, y)
        if error < best[1]:
            best = [model_dict[model], error]
    return best

# data1 includes our data without scaling or encoding and only uses these features: 'sqft', 'lotsqft', 'tax', 'age'
# data includes all features with scaling and encoding
# data2 eliminates features that did not pass our data validation test
# data3 is data with all cluster features added
# data3a is data3 but only amenities clusters
# data3n is data3 but only neighborhood clusters
# data4 is data3 but with feature validation
# data4a is data4 but only amenities clusters
# data3n is data4 but only neighborhood clusters
rem_neighbor = [c for c in X4_train.columns if c[0] == 'n']
rem_amenity = [c for c in X4_train.columns if c[0] == 'a']
insig = ['age', 'la', 'lotsqft', 'orange', 'pool', 'sqft', 'tax']
X3_train = X3_train.drop(columns='logerror')
datasets = {
    'data1': {'x': X1_train, 'y': y1_train, 'test': test1},
    'data': {'x': X_train.drop(columns='logerror'), 'y': y_train, 'test': test},
    'data2': {'x': X2_train, 'y': y_train, 'test': test2},
    'data3': {'x': X3_train, 'y': y_train, 'test': test3},
    'data3a': {'x': X3_train.drop(columns=neighbor_feats), 'y': y_train, 'test': test3.drop(columns=neighbor_feats)},
    'data3n': {'x': X3_train.drop(columns=amenities_feats), 'y': y_train, 'test': test3.drop(columns=amenities_feats)},
    'data4': {'x': X4_train, 'y': y_train, 'test': test4},
    'data4a': {'x': X4_train.drop(columns=rem_neighbor), 'y': y_train, 'test': test4.drop(columns=rem_neighbor)},
    'data4n': {'x': X4_train.drop(columns=rem_amenity), 'y': y_train, 'test': test4.drop(columns=rem_amenity)},
    'data5': {'x': X2_train.drop(columns=insig), 'y': y_train, 'test': test2.drop(columns=insig)}}

def many_models(dataset):
    data_x = dataset['x']
    data_y = dataset['y']
    models = {}

    lm2 = LinearRegression()
    lm2.fit(data_x, data_y)
    models['lm2'] = {'obj': lm2, 'data': dataset}

    lm3 = LinearSVR(random_state=123)
    lm3.fit(data_x, data_y)
    models['lm3'] = {'obj': lm3, 'data': dataset}

    lm4 = LinearSVR(random_state=123, dual=False, loss='squared_epsilon_insensitive')
    lm4.fit(data_x, data_y)
    models['lm4'] = {'obj': lm4, 'data': dataset}

    lm5 = SGDRegressor(random_state=123)
    lm5.fit(data_x, data_y)
    models['lm5'] = {'obj': lm5, 'data': dataset}

    lm6 = SGDRegressor(random_state=123, loss='huber')
    lm6.fit(data_x, data_y)
    models['lm6'] = {'obj': lm6, 'data': dataset}

    lm7 = SGDRegressor(random_state=123, loss='epsilon_insensitive')
    lm7.fit(data_x, data_y)
    models['lm7'] = {'obj': lm7, 'data': dataset}

    lm8 = SGDRegressor(random_state=123, loss='squared_epsilon_insensitive')
    lm8.fit(data_x, data_y)
    models['lm8'] = {'obj': lm8, 'data': dataset}

    lm9 = LassoCV()
    lm9.fit(data_x, data_y)
    models['lm9'] = {'obj': lm8, 'data': dataset}

    return models

best_ones = {}
for dataset in datasets:
    models = many_models(datasets[dataset])
    best_ones[dataset] = [best_model(models)[0]['obj'], best_model(models)[1]]

best_tests = {}
for dataset in datasets:
    models = many_models(datasets[dataset])
    best_tests[dataset] = [best_test(models)[0]['obj'], best_test(models)[1]]

set(zip(datasets['data5']['x'].columns, best_ones['data5'][0].coef_))