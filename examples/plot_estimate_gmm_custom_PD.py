"""
============================================
Estimate Gaussian Mixture Model from Samples
============================================

The maximum likelihood estimate (MLE) of a GMM cannot be computed directly.
Instead, we have to use expectation-maximization (EM). Then we can sample from
the estimated distribution or compute conditional distributions.
"""
print(__doc__)

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from gmr.utils import check_random_state
from gmr import GMM, plot_error_ellipses


UPPER_LIMIT = 3
LOWER_LIMIT = -3
NUM_GAUSS = 3


random_state = check_random_state(0)

n_samples = 300
n_features = 2
# X = np.ndarray((n_samples, n_features))
# X[:n_samples // 3, :] = random_state.multivariate_normal(
#     [0.0, 1.0], [[0.5, -1.0], [-1.0, 5.0]], size=(n_samples // 3,))
# X[n_samples // 3:-n_samples // 3, :] = random_state.multivariate_normal(
#     [-2.0, -2.0], [[3.0, 1.0], [1.0, 1.0]], size=(n_samples // 3,))
# X[-n_samples // 3:, :] = random_state.multivariate_normal(
#     [3.0, 1.0], [[3.0, -1.0], [-1.0, 1.0]], size=(n_samples // 3,))


data = loadmat('x_y_data')
x = list(map(lambda n: n[0], data['x_data']))
y = list(map(lambda n: n[0], data['y_data']))
X = np.array(zip(x, y))

gmm = GMM(n_components=NUM_GAUSS, random_state=1)
gmm.from_samples(X)
cond = gmm.condition(np.array([0]), np.array([1.0]))

# plt.figure(figsize=(15, 5))

# plt.subplot(1, 3, 1)
plt.title("Gaussian Mixture Model")
# plt.xlim((LOWER_LIMIT, UPPER_LIMIT))
# plt.ylim((LOWER_LIMIT, UPPER_LIMIT))
# plot_error_ellipses(plt.gca(), gmm, alpha=0.1, colors=["r", "g", "b"])
# plt.scatter(X[:, 0], X[:, 1])

# plt.subplot(1, 3, 2)
plt.title("Probability Density and Samples")
plt.xlim((LOWER_LIMIT, UPPER_LIMIT))
plt.ylim((LOWER_LIMIT, UPPER_LIMIT))
x, y = np.meshgrid(np.linspace(LOWER_LIMIT, UPPER_LIMIT, 100), np.linspace(LOWER_LIMIT, UPPER_LIMIT, 100))
X_test = np.vstack((x.ravel(), y.ravel())).T
p = gmm.to_probability_density(X_test)
p = p.reshape(*x.shape)
plt.contourf(x, y, p, colors=('#FFFFFF', '#FEE89F', '#D0F87C', '#58F5A9', '#19E5E3', '#41B0F1', '#3C7DE4', '#2B4BE6'))
X_sampled = gmm.sample(100)
s = [2] * 100
plt.scatter(X_sampled[:, 0], X_sampled[:, 1], c="pink", s=s)
plt.xlabel('x')
plt.ylabel('y')

# plt.subplot(1, 3, 3)
# plt.title("Conditional PDF $p(y | x = 1)$")
# X_test = np.linspace(LOWER_LIMIT, UPPER_LIMIT, 100)
# plt.plot(X_test, cond.to_probability_density(X_test[:, np.newaxis]))

plt.savefig('PD and Samples')
plt.show()
