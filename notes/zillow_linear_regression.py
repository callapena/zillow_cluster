import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from math import sqrt
import warnings
warnings.filterwarnings("ignore")

import env
import acquire
import prepare

zillow = acquire.get_zillow()
zillow = prepare.prep_zillow(zillow)
zillow = zillow[['logerror','sqft','fips','lat', 'long', 'lotsqft', 'age', 'tax']]
zillow.info()
zillow.age.value_counts(bins = 10).sort_index()


#### Home
# df['home_age'] = 2017 - df.yearbuilt
# df.home_age.value_counts().sort_index()

# df.info()
# df = df.drop(columns = ['yearbuilt'])

# df.set_index('parcelid', inplace = True)

sns.distplot(df.logerror)
plt.xlim(-1,1)
plt.ylim(0,8)
sns.distplot(df.sqft)

df.sqft.value_counts(dropna=False)

train, test = train_test_split(zillow, test_size = .3, random_state = 123)

train.info()
train.sqft.value_counts(dropna=False)
train = train.fillna(0)
train.info()

train.corr()

X = train[['sqft', 'lotsqft', 'tax', 'age']]
y = train[['logerror']]

X_test = test[['sqft', 'lotsqft', 'tax', 'age']]
y_test = test[['logerror']]

from sklearn.linear_model import LinearRegression
X.info()

lm1 = LinearRegression()

lm1.fit(X, y)

y_pred_lm1 = lm1.predict(X)

y_test_pred_lm1 = lm1.predict(X_test)



###Scores
mean_squared_error(y, y_pred_lm1)**1/2
mean_squared_error(y_test, y_test_pred_lm1)**1/2


r2_score(y, y_pred_lm1)

y.index.values.reshape(-1,1)

y_pred_baseline = np.array([y.mean()[0]]*len(y))


model = pd.DataFrame({'actual': y.logerror,
              'lm1': y_pred_lm1.ravel(),
              'y_baseline': y_pred_baseline.ravel(),
              'fips': X.fips})

sns.relplot(x = 'lm1', y='actual', data=model)
sns.relplot(x = 'lm1', y='y_baseline', data=model, color = 'red')
plt.show()

datasets = {
    'data1': {'x': X, 'y': y, 'test': X_test}}

datasets['data1']['x']

data_x = dataset['x']
data_y = dataset['y']
models = {}

lm2 = LinearRegression()
lm2.fit(data_x, data_y)
models['lm2'] = {'obj': lm2, 'data': dataset}



