# import numpy as np
# from sklearn.decomposition import PCA
# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# pca = PCA(n_components=2)
# pca.fit(X)
# print(pca.explained_variance_ratio_)


# import numpy as np
#
# # Приклад матриці
# # matrix_a = np.array([[1, 2, 3],
# #                     [4, 5, 6]])
#
# matrix_a = np.array([[4, 0],
#                     [3, -5]])
#
# # Обчислення транспонованої матриці
# matrix_a_transposed = np.transpose(matrix_a)  # або matrix_a.T
#
# # Обчислення добутку матриці на її транспоновану
# result = np.dot(matrix_a, matrix_a_transposed) # або matrix_a @ matrix_a_transposed
#
# # Виведення результату
# print("Матриця A:")
# print(matrix_a)
#
# print("\nТранспонована матриця A:")
# print(matrix_a_transposed)
#
# print("\nДобуток матриці A на її транспоновану:")
# print(result)
#
# # Обчислення власних векторів та власних значень
# eigenvalues, eigenvectors = np.linalg.eig(result)
#
# # Виведення результату
# print("Власні значення:")
# print(eigenvalues)
#
# print("\nВласні вектори:")
# print(eigenvectors)
#
#
# # Отримання індексів, які відсортовують масив за першим стовпцем
# sorted_indices = np.argsort(eigenvectors[:, 0])
#
# V = eigenvectors[sorted_indices]
#
# # Виведення результату
# print("Вихідний масив:")
# print(eigenvectors)
#
# print("\nМатриця V:")
# print(V)
#
# # Створення діагонального масиву
# Sigma = np.diag(sorted(eigenvalues, reverse=True))
# Sigma = np.sqrt(Sigma)
# print("\nМатриця Σ:")
# print(Sigma)
#
# Sigma.transpose()
#
# AV = np.dot(matrix_a, V)
#
# def matrix_norm(mtr):
#
#     # Обчислення кореня квадратного з суми елементів для кожного стовпця
#     sqrt_sum_columns = np.sqrt(np.sum(mtr**2, axis=0))
#     res = mtr / sqrt_sum_columns
#     return res
#
# AV = matrix_norm(AV)
# print()
# print(AV)
#
# U = np.dot(AV, Sigma.T)
# U = matrix_norm(U)
# print('Матриця U')
# print(U)
#
# A = np.dot(np.dot(U, Sigma), V.T)
# print('Початкова матриця A')
# print(A)
#
# print(U)
# print(Sigma)
# print(V)
#
# Uu, Ss, Vh = np.linalg.svd(matrix_a)
#
# print('SVD U ')
# print(Uu)
# print('SVD Sigma ')
# print(Ss)
# print('SVD Vh ')
# print(Vh)


3.

import numpy as np
#
# A = np.array([[1, 2, 3],
#               [4, -1, 0],
#               [-2, 5, 1]])
#
# E1 = np.array([[1,  0, 0],
#                [-4, 1, 0],
#                [0,  0, 1]])
#
# E2 = np.array([[1, 0, 0],
#                [0, 1, 0],
#                [2, 0, 1]])
#
# E3 = np.array([[1, 0, 0],
#                [0, 1, 0],
#                [0, 1, 1]])
#
# E1_inverse = np.linalg.inv(E1)
# E2_inverse = np.linalg.inv(E2)
# E3_inverse = np.linalg.inv(E3)
#
# U = E3.dot(E2).dot(E1).dot(A)
# L = E1_inverse.dot(E2_inverse).dot(E3_inverse)
#
# print("\nStep 1 & 2: Upper traingular matrix of A using elementary matrices:")
# print(U)
# print("\nStep 1 & 3: Lower traingular matrix of A using inverse elementary matrices:")
# print(L)
#
# U_inverse = np.linalg.inv(U)
# L_inverse = np.linalg.inv(L)
#
# b1 = np.array([[3],
#                [9],
#                [-8]]) # column vector
#
# c1 = L_inverse.dot(b1)
# x1 = U_inverse.dot(c1)
# print("\nStep 4a: Solve c1 given same left hand side matrix A but different right hand side b1:")
# print(c1)
# print("\nStep 5b: Solution x1 given same left hand side matrix A but different right hand side b1:")
# print(x1)
#
# b2 = np.array([[28],
#                [22],
#                [-11]]) # column vector
#
# c2 = L_inverse.dot(b2)
# x2 = U_inverse.dot(c2)
# print("\nStep 4a: Solve c2 given same left hand side matrix A but different right hand side b2:")
# print(c2)
# print("\nStep 5b: Solution x2 given same left hand side matrix A but different right hand side b2:")
# print(x2)

4.

# import pprint # In order to print matrices prettier
# import scipy as sc
# import scipy.linalg # Linear Algebra Library contained in Scipy
#
#
# matrix_A = sc.array([[7,4],[3,5]]) # given matrix A
# P, L, U = scipy.linalg.lu(matrix_A) # returns the result of LU decomposition to the variables P, L, and U
#
# print("Original matrix A:")
# pprint.pprint(matrix_A)
#
# # implies a pivoting(reordering) rows(or columns) in case it is needed
# print("Pivoting matrix P:")
# pprint.pprint(P)
#
# # lower-triangular matrix of A
# print("L:")
# pprint.pprint(L)
#
# # upper-triangular matrix of A
# print("U:")
# pprint.pprint(U)
5.

# from numpy import array
# from numpy.linalg import cholesky
# # define a 3x3 matrix
# A = array([[36, 30, 18], [30, 41, 23], [18, 23, 14]])
# print(A)
# # Cholesky decomposition
# L = cholesky(A)
# print()
# print(L)
# print()
# print(L.T)
# # reconstruct
# B = L.dot(L.T)
# print()
# print(B)

6.
