import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from prepare import prep_zillow

import seaborn as sns
sns.set_style('whitegrid')
sns.set_palette('husl')

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from mpl_toolkits.mplot3d import Axes3D

zillow = pd.read_csv('zillow.csv')
zillow = prep_zillow(zillow)

train, test = train_test_split(zillow, random_state = 123)

ks = range(1,10)
sse = []
for k in ks:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(train[['logerror']])

    # inertia: Sum of squared distanes of samplesto their closest cluster
    sse.append(kmeans.inertia_)
print(pd.DataFrame(dict(k=ks, sse=sse)))

plt.plot(ks, sse, 'bx-')
plt.xlabel('k')
plt.ylabel('SSE')
plt.title('The Elbow Method to find the optimal k')
plt.show()

kmeans = KMeans(n_clusters = 4)
kmeans.fit(train_vars[['logerror']])

train['cluster_4'] = kmeans.predict(train[['logerror']])
train.cluster_4.value_counts()
train.cluster_4 = 'cluster_' + train_vars.cluster_4.astype('str')


train.corr().logerror

