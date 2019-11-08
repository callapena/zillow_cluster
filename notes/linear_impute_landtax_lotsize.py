import acquire
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# df = acquire.get_zillow()
# df.info()

# sns.scatterplot(x='landtaxvaluedollarcnt', y='lotsizesquarefeet', data=df)
# plt.yscale('log')

# sns.scatterplot(x = 'calculatedfinishedsquarefeet', y = 'logerror', data=df)

# sns.scatterplot(x='landtaxvaluedollarcnt', y='logerror', data=df)

def linear_impute(df, x, y):
    rowdrop = df[(df[x].isna()==True) | (df[y].isna()==True)].index
    land_lot_df = df.drop(rowdrop)[[x, y]]
    lm1 = LinearRegression()
    lm1.fit(land_lot_df[[x]], land_lot_df[[y]])
    X = df[(df[y].isna()==True)][[x]]
    y_hat = pd.DataFrame(lm1.predict(X),columns = ['yhat']).set_index(X.index.values).yhat
    df[y] = df[y].fillna(y_hat)
    return df

# df = linear_impute(df)
# df['lotsizesquarefeet'][50]
# df['lotsizesquarefeet'][79]

# sns.scatterplot(x='landtaxvaluedollarcnt', y='lotsizesquarefeet', data=df)
# plt.yscale('log')