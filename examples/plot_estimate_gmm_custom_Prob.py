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
from matplotlib import collections  as mc



UPPER_LIMIT = 3
LOWER_LIMIT = -3
NUM_GAUSS = 3
N = 100.0
STEP = (UPPER_LIMIT - LOWER_LIMIT)/N


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


# plt.figure(figsize=(15, 5))

# plt.subplot(1, 3, 1)
# plt.title("Gaussian Mixture Model")
# plt.xlim((LOWER_LIMIT, UPPER_LIMIT))
# plt.ylim((LOWER_LIMIT, UPPER_LIMIT))
# plot_error_ellipses(plt.gca(), gmm, alpha=0.1, colors=["r", "g", "b"])
# plt.scatter(X[:, 0], X[:, 1])

# plt.subplot(1, 3, 2)

# lines = [[(0, 1), (1, 1)], [(2, 3), (3, 3)], [(1, 2), (1, 3)]]
lines = []
prev = None
# c = np.array([(1, 1, 1, 1), (0, 1, 0, 1), (0, 0, 1, 1)])
c = []

X_sampled = gmm.sample(100)
SAMPLES = 50
count = 0
for i in np.linspace(-3,4, SAMPLES):
    plt.figure(figsize=(6, 5))

    ax = plt.subplot(1, 1, 1)
    plt.title("Probability Density and Samples")
    plt.xlim((LOWER_LIMIT, UPPER_LIMIT))
    plt.ylim((LOWER_LIMIT, UPPER_LIMIT))
    x, y = np.meshgrid(np.linspace(LOWER_LIMIT, UPPER_LIMIT, 100), np.linspace(LOWER_LIMIT, UPPER_LIMIT, 100))
    X_test = np.vstack((x.ravel(), y.ravel())).T
    p = gmm.to_probability_density(X_test)
    p = p.reshape(*x.shape)
    plt.contourf(x, y, p, colors=('#FFFFFF', '#FEE89F', '#D0F87C', '#58F5A9', '#19E5E3', '#41B0F1', '#3C7DE4', '#2B4BE6'))
    s = [2] * 100
    plt.scatter(X_sampled[:, 0], X_sampled[:, 1], c="pink", s=s)
    plt.xlabel('x')
    plt.ylabel('y')

    interception_x = i
    # plt.axvline(x=interception_x, c='r')

    X_test = np.linspace(LOWER_LIMIT, UPPER_LIMIT, 100)
    cond = gmm.condition(np.array([0]), np.array([i]))
    y = cond.to_probability_density(X_test[:, np.newaxis])
    # print(np.argmax(y)*STEP + LOWER_LIMIT)
    interception_y = np.argmax(y)*STEP + LOWER_LIMIT
    # plt.axhline(y=interception_y, c='r')

    if prev:
        curr = (interception_x, interception_y)
        lines.append([prev, curr])
        # print(lines)
        prev = curr

        c = [(0, 0, 0)] * count
        c = np.array(c)
        lc = mc.LineCollection(lines, colors=c, linewidths=2)
        # fig, ax = plt.subplots()
        ax.add_collection(lc)
    else:
        prev = (interception_x, interception_y)
        
    
    # ax = plt.subplot(1, 2, 2)
    # title = "Conditional PDF $p(y | x = " + str(round(i, 2)) + ")$"
    # plt.title(title)
    # # plt.plot(X_test, y)
    # # plt.xticks(np.arange(0, 1.0, 0.1))
    # plt.xlim(0, 0.9)
    # # ax.xaxis.set_ticks(np.arange(0, 1.0, 0.1))
    # plt.plot(y, X_test)
    # plt.xlabel('x')
    # plt.ylabel('y')
    plt.savefig('Prob/Prob. ' + str(count) + '.png')
    count += 1
    # plt.show()

