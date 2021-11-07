"""
=======================
Clustering Iris Dataset
=======================

The Iris dataset is a typical classification problem. We will cluster
it with GMM and see whether the clusters roughly match the classes.
For better visualization will will first reduce the dimensionality to
two with PCA from sklearn. We will also use sklearn to load the dataset.

We display samples by dots with colors indicating their true class.
The clusters are represented by their error ellipses.
"""
print(__doc__)
from scipy.io import loadmat
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from gmr import GMM, plot_error_ellipses
import numpy as np

data = loadmat('x_y_data')
x = list(map(lambda n: n[0], data['x_data']))
y = list(map(lambda n: n[0], data['y_data']))
X_pca = np.array(zip(x, y))

# X, y = load_iris(return_X_y=True)
# X_pca = PCA(n_components=2, whiten=True, random_state=0).fit_transform(X)

gmm = GMM(n_components=3, random_state=1)
gmm.from_samples(X_pca)

plt.figure()
ax = plt.subplot(111)
ax.scatter(x, y)
plot_error_ellipses(ax, gmm, alpha=0.1, colors=["r", "g", "b"])
plt.show()