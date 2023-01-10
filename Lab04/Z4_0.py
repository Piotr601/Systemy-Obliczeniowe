import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import DistanceMetric
import matplotlib.pyplot as plt

NUM_GROUPS = 7

# Tworzenie datasetu z dwoma cechami informatywnymi
X, y = make_classification(n_samples=10000, n_features=2, n_informative=2, n_redundant=0, random_state=1410)

# Wybieranie losowych instacji problemu
centroids = X[np.random.choice(X.shape[0], size=NUM_GROUPS, replace=False)]
dist_metric = DistanceMetric.get_metric('euclidean')

prev_center = np.zeros(np.shape(centroids))
clusters = np.zeros(X.shape[0])
distances = np.zeros((X.shape[0], NUM_GROUPS))

while True:
    distances = dist_metric.pairwise(X, centroids)
    clusters = np.argmin(distances, axis=1)
    prev_center = np.copy(centroids)
    
    for i in range(NUM_GROUPS):
        centroids[i] = np.mean(X[clusters == i], axis=0)
    
    if (np.allclose(prev_center, centroids)) == False: break

plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', edgecolor="k", s=80)
plt.show()