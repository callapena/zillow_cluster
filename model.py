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

if __name__ == '__main__':
    zillow = get_zillow()
    zillow = prep_zillow(zillow)