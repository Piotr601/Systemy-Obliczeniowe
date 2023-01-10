import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import DistanceMetric
from mpi4py import MPI


# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

groups = 3
X, y = make_classification(n_samples=10000, n_features=2, n_informative=2, n_redundant=0, random_state=1410)

Xcopy = X

clusters = np.zeros(X.shape[0])
distances = np.zeros((X.shape[0], 3))
dist_metric = DistanceMetric.get_metric('euclidean')

print('Running')
if rank == 0:
    centroids = X[np.random.choice(np.shape(X)[0], size=groups, replace=False)]
    
    ave, res = divmod(np.shape(X)[0], size)
    counts = [ave + 1 if p < res else ave for p in range(size)]
    starts = [sum(counts[:p]) for p in range(size)]
    ends = [sum(counts[:p+1]) for p in range(size)]
    X = [X[starts[p]:ends[p]] for p in range(size)]
else:
    X = None
    centroids = None


X_d = comm.scatter(X, root=0)
centroids = comm.bcast(centroids, root=0)

print(f"P: {rank}, recieved data: \n{X_d}, \ncentroids: \n{centroids}, \ninstantion: \n{clusters}")

for i in range(groups):
    distances = dist_metric.pairwise(X_d, centroids)
    clusters = np.argmin(distances, axis=1)
    
    centroids[i] = np.mean(X_d[clusters==i], axis=0)
    n_centroids = np.random.choice(np.shape(X_d)[0], size=1)
    
print(f"P: {rank}, {clusters}, {n_centroids}")

membership = np.argmin(distances, axis=0)

X_all = comm.gather(X_d, root=0)
membership_all = comm.gather(centroids, root=0)

print(f'P: {rank}, \nX_all: {X_all}, \nMembership: {membership_all}')

if rank == 0:
    plt.scatter(Xcopy[:, 0], Xcopy[:, 1], cmap='viridis')
    plt.scatter(centroids[:,0], centroids[:,1], c='r', s=80)
    plt.show()