import numpy as np

1.
#
# import numpy as np
# from numpy import linalg as LA
#
# input = np.array([[2,2],[8,2]])
#
# eig_val, eig_vect = LA.eig(input)
#
# print(eig_val)
# print(eig_vect)

2.
A = np.array([
  [0, 1, 1, 0, 0, 0, 0, 0, 1, 1],
  [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
  [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
  [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
  [0, 0, 0, 1, 1, 0, 1, 1, 0, 0],
  [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
  [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
  [1, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
#
# D = np.diag(A.sum(axis=1))
# print(D)
#
# # [[4 0 0 0 0 0 0 0 0 0]
# #  [0 2 0 0 0 0 0 0 0 0]
# #  [0 0 2 0 0 0 0 0 0 0]
# #  [0 0 0 2 0 0 0 0 0 0]
# #  [0 0 0 0 2 0 0 0 0 0]
# #  [0 0 0 0 0 4 0 0 0 0]
# #  [0 0 0 0 0 0 2 0 0 0]
# #  [0 0 0 0 0 0 0 2 0 0]
# %%
# #  [0 0 0 0 0 0 0 0 2 0]
# %%


# #  [0 0 0 0 0 0 0 0 0 2]]
#
# L = D-A
# print(L)
#
# # [[ 4 -1 -1  0  0  0  0  0 -1 -1]
# #  [-1  2 -1  0  0  0  0  0  0  0]
# #  [-1 -1  2  0  0  0  0  0  0  0]
# #  [ 0  0  0  2 -1 -1  0  0  0  0]
# #  [ 0  0  0 -1  2 -1  0  0  0  0]
# #  [ 0  0  0 -1 -1  4 -1 -1  0  0]
# #  [ 0  0  0  0  0 -1  2 -1  0  0]
# #  [ 0  0  0  0  0 -1 -1  2  0  0]
# #  [-1  0  0  0  0  0  0  0  2 -1]
# #  [-1  0  0  0  0  0  0  0 -1  2]]

3.

# from sklearn.cluster import KMeans
#
# # our adjacency matrix
# print("Adjacency Matrix:")
# print(A)
#
# # Adjacency Matrix:
# # [[0. 1. 1. 0. 0. 1. 0. 0. 1. 1.]
# #  [1. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
# #  [1. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 0. 0. 1. 1. 0. 0. 0. 0.]
# #  [0. 0. 0. 1. 0. 1. 0. 0. 0. 0.]
# #  [1. 0. 0. 1. 1. 0. 1. 1. 0. 0.]
# #  [0. 0. 0. 0. 0. 1. 0. 1. 0. 0.]
# #  [0. 0. 0. 0. 0. 1. 1. 0. 0. 0.]
# #  [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
# #  [1. 0. 0. 0. 0. 0. 0. 0. 1. 0.]]
#
# # diagonal matrix
# D = np.diag(A.sum(axis=1))
#
# # graph laplacian
# L = D-A
#
# # eigenvalues and eigenvectors
# vals, vecs = np.linalg.eig(L)
#
# # sort these based on the eigenvalues
# vecs = vecs[:,np.argsort(vals)]
# vals = vals[np.argsort(vals)]
#
# # kmeans on first three vectors with nonzero eigenvalues
# kmeans = KMeans(n_clusters=4)
# kmeans.fit(vecs[:,1:4])
# colors = kmeans.labels_
#
# print("Clusters:", colors)
#
# # Clusters: [2 1 1 0 0 0 3 3 2 2]

4.
 
# from sklearn.datasets import make_circles
# from sklearn.neighbors import kneighbors_graph
# import numpy as np
#
# # create the data
# X, labels = make_circles(n_samples=500, noise=0.1, factor=.2)
#
# # use the nearest neighbor graph as our adjacency matrix
# A = kneighbors_graph(X, n_neighbors=5).toarray()
# print(A)
#
# # [[0. 0. 0. ... 0. 0. 0.]
# #  [0. 0. 0. ... 0. 0. 0.]
# #  [0. 0. 0. ... 0. 0. 0.]
# #  ...
# #  [0. 0. 0. ... 0. 1. 0.]
# #  [0. 0. 0. ... 0. 0. 0.]
# #  [0. 0. 0. ... 0. 0. 0.]]
#
# # create the graph laplacian
# D = np.diag(A.sum(axis=1))
# L = D-A
#
# # find the eigenvalues and eigenvectors
# vals, vecs = np.linalg.eig(L)
#
# # sort
# vecs = vecs[:,np.argsort(vals)]
# vals = vals[np.argsort(vals)]
#
# # use Fiedler value to find best cut to separate data
# clusters = vecs[:,1] > 0
5.
