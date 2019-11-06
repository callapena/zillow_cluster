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

df = acquire.get_zillow()
df = df[['logerror', 'parcelid','calculatedfinishedsquarefeet','fips','latitude', 'longitude', 'lotsizesquarefeet', 'yearbuilt', 'taxvaluedollarcnt']]
df.info()
df.yearbuilt.value_counts().sort_index()

#### Home
df['home_age'] = 2017 - df.yearbuilt
df.home_age.value_counts().sort_index()

df.info()
df = df.drop(columns = ['yearbuilt'])

df.set_index('parcelid', inplace = True)

sns.distplot(df.logerror)
sns.distplot(df.calculatedfinishedsquarefeet)

df.calculatedfinishedsquarefeet.value_counts(dropna=False)
df.dropna(axis = 0)

train, test = train_test_split(df, random_state = 123)

train.info()
train.calculatedfinishedsquarefeet.value_counts(dropna=False)
train = train.fillna(0)
train.info()

train.corr()

X = train[['calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'taxvaluedollarcnt', 'home_age', 'fips']]
y = train[['logerror']]

from sklearn.linear_model import LinearRegression

lm1 = LinearRegression()

lm1.fit(X, y)

y_pred_lm1 = lm1.predict(X)


###Scores
mean_squared_error(y, y_pred_lm1)

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