1.
# # Import Numpy package and the norm function
# import numpy as np
# from numpy.linalg import norm
#
#
# # Define a vector
# v = np.array([2,3,1,0])
#
# # Take the q-norm which p=2
# p = 2
# v_norm = norm(v, ord=p)
#
# # Print values
# print('The vector: ', v)
# print('The vector norm: ', v_norm)
2.
# # max norm of a vector
# from numpy import inf
# from numpy import array
# from numpy.linalg import norm
# v = array([1, 2, 3])
# print(v)
# maxnorm = norm(v, inf)
# print(maxnorm)

3.
# from math import sqrt
# import numpy as np
#
#
# # Function to return the Frobenius
# # Norm of the given matrix
# def frobeniusNorm(mat):
#     row = np.shape(mat)[0]
#     col = np.shape(mat)[1]
#     # To store the sum of squares of the
#     # elements of the given matrix
#     sumSq = 0
#     for i in range(row):
#         for j in range(col):
#             sumSq += pow(mat[i][j], 2)
#
#     # Return the square root of
#     # the sum of squares
#     res = sqrt(sumSq)
#     return round(res, 5)
#
#
# # Driver code
#
#
# mat = [[1, 2, 3], [4, 5, 6]]
#
# print(frobeniusNorm(mat))

4.
# import numpy as np
#
# # Задаємо координати точок у двовимірному просторі
# point1 = np.array([1, 2])
# point2 = np.array([4, 6])
#
# # Відстань Евкліда між точками
# euclidean_distance = np.linalg.norm(point1 - point2)
#
# print(f"Координати точки 1: {point1}")
# print(f"Координати точки 2: {point2}")
# print(f"Відстань Евкліда між точками: {euclidean_distance}")

5.
# from scipy.spatial.distance import cityblock
# import pandas as pd
#
# #define DataFrame
# df = pd.DataFrame({'A': [2, 4, 4, 6],
#                    'B': [5, 5, 7, 8],
#                    'C': [9, 12, 12, 13]})
#
# #calculate Manhattan distance between columns A and B
# cityblock(df.A, df.B)
#
# print(cityblock(df.A, df.B))
6.
# from scipy.spatial import distance
# distance.chebyshev([1, 0, 0], [0, 1, 0])
#
# distance.chebyshev([1, 1, 0], [0, 1, 0])
#
# print(distance.chebyshev([1, 1, 0], [0, 1, 0]))
7.
# from scipy.spatial import distance
# distance.minkowski([1, 0, 0], [0, 1, 0], 1)
#
# distance.minkowski([1, 0, 0], [0, 1, 0], 2)
#
# distance.minkowski([1, 0, 0], [0, 1, 0], 3)
#
# distance.minkowski([1, 1, 0], [0, 1, 0], 1)
#
# distance.minkowski([1, 1, 0], [0, 1, 0], 2)
#
# distance.minkowski([1, 1, 0], [0, 1, 0], 3)
#
# print(distance.minkowski([1, 1, 0], [0, 1, 0], 3))

8.

# from scipy.spatial import distance
# distance.cosine([1, 0, 0], [0, 1, 0])
#
# distance.cosine([100, 0, 0], [0, 1, 0])
#
# distance.cosine([1, 1, 0], [0, 1, 0])
#
# print(distance.cosine([1, 1, 0], [0, 1, 0]))
9.
# from scipy.spatial import distance
# distance.jaccard([1, 0, 0], [0, 1, 0])
# 1.0
# distance.jaccard([1, 0, 0], [1, 1, 0])
# 0.5
# distance.jaccard([1, 0, 0], [1, 2, 0])
# 0.5
# distance.jaccard([1, 0, 0], [1, 1, 1])
# print(distance.jaccard([1, 0, 0], [1, 1, 1]))

10.

# from scipy.spatial import distance
# distance.dice([1, 0, 0], [0, 1, 0])
#
# distance.dice([1, 0, 0], [1, 1, 0])
#
# distance.dice([1, 0, 0], [2, 0, 0])
#
# print(distance.dice([1, 0, 0], [2, 0, 0]))
11.
# from Levenshtein import distance as lev
#
# #calculate Levenshtein distance
# lev('party', 'park')
# print(lev('party', 'park'))

12.
# from scipy.spatial.distance import hamming
#
# #define arrays
# x = ['a', 'b', 'c', 'd']
# y = ['a', 'b', 'c', 'r']
#
# #calculate Hamming distance between the two arrays
# hamming(x, y) * len(x)
#
# print(hamming(x, y) * len(x))
13.
# from haversine import haversine, Unit
#
# lyon = (45.7597, 4.8422)  # (lat, lon)
# paris = (48.8567, 2.3508)
#
# print(haversine(lyon, paris), 'in kilometers')
#
# print(haversine(lyon, paris, unit=Unit.MILES), 'in miles')
#
# # you can also use the string abbreviation for units:
# print(haversine(lyon, paris, unit='mi'), 'in miles')
#
# print(haversine(lyon, paris, unit=Unit.NAUTICAL_MILES), 'in nautical miles')

14.
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import Normalizer
#
#
# def data_scale(data, scaler_type='minmax'):
#     if scaler_type == 'minmax':
#         scaler = MinMaxScaler()
#     if scaler_type == 'std':
#         scaler = StandardScaler()
#     if scaler_type == 'norm':
#         scaler = Normalizer()
#
#     scaler.fit(data)
#     res = scaler.transform(data)
#     return res
#
#
# print(data_scale([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

15.

import numpy as np
from scipy.stats import pearsonr

# Згенеруємо дві змінні з кореляцією
np.random.seed(42)
X = np.random.rand(100)
Y = 2 * X + 1 + 0.1 * np.random.randn(100)

# Розрахунок коефіцієнта кореляції Пірсона
correlation_coefficient, _ = pearsonr(X, Y)

print(f"Коефіцієнт кореляції Пірсона: {correlation_coefficient}")
16.
from scipy.stats import spearmanr

# Використовуємо ті самі дані X і Y

# Розрахунок коефіцієнта кореляції Спірмена
spearman_coefficient, _ = spearmanr(X, Y)

print(f"Коефіцієнт кореляції Спірмена: {spearman_coefficient}")

from scipy.stats import kendalltau

# Використовуємо ті самі дані X і Y

# Розрахунок коефіцієнта кореляції Кендала
kendall_coefficient, _ = kendalltau(X, Y)

print(f"Коефіцієнт кореляції Кендала: {kendall_coefficient}")