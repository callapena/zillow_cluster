import acquire
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


df = acquire.get_zillow()
df.info()

sns.scatterplot(x='landtaxvaluedollarcnt', y='lotsizesquarefeet', data=df)
plt.yscale('log')

sns.scatterplot(x = 'calculatedfinishedsquarefeet', y = 'logerror', data=df)

sns.scatterplot(x='landtaxvaluedollarcnt', y='logerror', data=df)

def linear_impute(df):
    rowdrop = df[(df.landtaxvaluedollarcnt.isna()==True) | (df.lotsizesquarefeet.isna()==True)].index.values
    land_lot_df = df.drop(rowdrop)[['landtaxvaluedollarcnt','lotsizesquarefeet']]
    lm1 = LinearRegression()
    lm1.fit(land_lot_df[['landtaxvaluedollarcnt']], land_lot_df[['lotsizesquarefeet']])
    X = df[(df.lotsizesquarefeet.isna()==True)][['landtaxvaluedollarcnt']]
    y_hat = pd.DataFrame(lm1.predict(X),columns = ['yhat']).set_index(X.index.values)
    y_hat = y_hat.yhat
    df['lotsizesquarefeet'] = df.lotsizesquarefeet.fillna(y_hat)
    return df

df = linear_impute(df)
df['lotsizesquarefeet'][50]
df['lotsizesquarefeet'][79]

sns.scatterplot(x='landtaxvaluedollarcnt', y='lotsizesquarefeet', data=df)
plt.yscale('log')
