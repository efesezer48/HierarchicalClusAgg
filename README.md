# Hierarchical Clustering - Agglomerative

Hierarchical clustering is an unsupervised learning method for clustering data points. The algorithm builds clusters by measuring the dissimilarities between data. Unsupervised learning means that a model does not have to be trained, and we do not need a "target" variable. This method can be used on any data to visualize and interpret the relationship between individual data points.

Here we will use hierarchical clustering to group data points and visualize the clusters using both a dendrogram and scatter plot.

Import the modules 

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

Create arrays that resemble two variables in a dataset. Note that while we only use two variables here, this method will work with any number of variables:
x = np.array([1, 2, 3, 4, 5, 6, 7, 11, 7, 15])
y = np.array([1, 4, 9, 6, 5, 2, 8, 10, 14, 10])

Turn the data into a set of points:

data = list(zip(x, y))
print(data)

Result:
[(1, 1), (2, 4), (3, 9), (4, 6), (5, 5), (6, 2), (7, 8), (11, 10), (7, 14), (15, 10)]

Compute the linkage between all of the different points. Here we use a simple euclidean distance measure and Ward's linkage, which seeks to minimize the variance between clusters.

linkage_data = linkage(data, method='ward', metric='euclidean')

Finally, plot the results in a dendrogram. This plot will show us the hierarchy of clusters from the bottom (individual points) to the top (a single cluster consisting of all data points).

plt.show() lets us visualize the dendrogram instead of just the raw linkage data.

dendrogram(linkage_data)
plt.show()

The scikit-learn library allows us to use hierarchichal clustering in a different manner. First, we initialize the AgglomerativeClustering class with 2 clusters, using the same euclidean distance and Ward linkage.

hierarchical_cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')

The .fit_predict method can be called on our data to compute the clusters using the defined parameters across our chosen number of clusters.

labels = hierarchical_cluster.fit_predict(data) print(labels)

RESULT: 
[0 0 0 0 0 0 0 1 1 1]
