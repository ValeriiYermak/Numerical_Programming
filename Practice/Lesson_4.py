import numpy as np
from scipy.linalg._expm_frechet import vec

# vec_c = np.cross([4, -5, 0], [0,4, -3])
#
# np.linalg.norm(vec_c)
# print(vec_c)

2.
# import numpy as np
#
# def normal_vector(u, v):
#     # Обчислення векторного добутку (cross product)
#     n = np.cross(u, v)
#
#     return n
#
# # Приклад використання
# u = np.array([1, 2, 3])
# v = np.array([4, 5, 6])
#
# normal = normal_vector(u, v)
# print("Вектор нормалі до площини:", normal)

# 3.
#
# import numpy as np
#
# def parallelepiped_volume(a, b, c):
#     # Обчислення мішаного добутку
#     mixed_dot_product = np.dot(a, np.cross(b, c))
#
#     # Обчислення об'єму паралелепіпеда
#     volume = abs(mixed_dot_product)
#
#     return volume
#
# # Приклад використання
# a = np.array([2, -2, -3])
# b = np.array([4, 0, 6])
# c = np.array([-7, -7, 1])
#
# volume = parallelepiped_volume(a, b, c)
# print("Об'єм паралелепіпеда:", volume)

4.
# import numpy as np
#
# def are_vectors_linearly_independent(vectors):
#     # Створення розширеної матриці з векторів
#     matrix = np.array(vectors).T
#
#     # Ранг матриці
#     rank_matrix = np.linalg.matrix_rank(matrix)
#     print(f'Rang matrix: {rank_matrix}')
#     # Кількість векторів
#     num_vectors = len(vectors)
#     print(f'Number of vectors: {num_vectors}')
#
#     # Вектори лінійно незалежні, якщо ранг матриці рівний кількості векторів
#     return rank_matrix == num_vectors
#
# # Приклад використання
# # vectors1 = np.array([1, 2, 3])
# # vectors2 = np.array([-2, 1, -1])
# # vectors3 = np.array([3, 2, -1])
#
# # Приклад використання
# vectors1 = np.array([1, 2, -3])
# vectors2 = np.array([-1, 2, 4])
# vectors3 = np.array([1, 6, -2])
#
# # Перевірка лінійної незалежності векторів
# result = are_vectors_linearly_independent([vectors1, vectors2, vectors3])
#
# if result:
#     print("Вектори лінійно незалежні.")
# else:
#     print("Вектори лінійно залежні.")

5.
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Функція для перетворення вектора у новий базис
# def transform_to_new_basis(vector, transformation_matrix):
#     return np.dot(transformation_matrix, vector)
#
# # Зображення (вектор у старому базисі)
# image_vector = np.array([3, 2])
#
# # Матриця перетворення (новий базис)
# transformation_matrix = np.array([[1, -1],
#                                    [1, 1]])
#
# # Перетворення вектора у новий базис
# transformed_vector = transform_to_new_basis(image_vector, transformation_matrix)
#
# print("Оригінальні вектори:")
# print(image_vector)
# print("\nМатриця перетворення:")
# print(transformation_matrix)
# print("\nВектори у новому базисі:")
# print(transformed_vector)
#
#
# # Зображення у старому та новому базисі
# plt.figure(figsize=(8, 4))
#
# # Зображення у старому базисі
# plt.subplot(1, 2, 1)
# plt.quiver(0, 0, image_vector[0], image_vector[1], angles='xy', scale_units='xy', scale=1, color='b')
# plt.title('Старий базис')
#
# # Зображення у новому базисі
# plt.subplot(1, 2, 2)
# plt.quiver(0, 0, transformed_vector[0], transformed_vector[1], angles='xy', scale_units='xy', scale=1, color='r')
# plt.title('Новий базис')
#
# plt.tight_layout()
# plt.show()

6.
# import numpy as np
#
# # Приклад: Обчислення відстані між двома точками
# point1 = np.array([1, 2])
# point2 = np.array([4, 6])
#
# distance = np.linalg.norm(point2 - point1)
# print(f"Відстань між точкою 1 та точкою 2: {distance}")
#
# # Приклад: Обчислення відстані між векторами
# vector1 = np.array([3, 5, 2])
# vector2 = np.array([1, 8, 4])
#
# distance_vector = np.linalg.norm(vector2 - vector1)
# print(f"Відстань між вектором 1 та вектором 2: {distance_vector}")
# 7.
#
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# # Розміри площини
# width = 10
# height = 10
#
# # Створення площини
# x = np.linspace(0, width, 100)
# y = np.linspace(0, height, 100)
# x, y = np.meshgrid(x, y)
#
# # Параметри джерела світла та спостерігача
# light_position = np.array([3, 3, 10])
# observer_position = np.array([2, 2, 0])
#
# # Вектори нормалей до площини (зазвичай вони всі спрямовані вгору від площини)
# normals = np.array([0, 0, 1])
#
# # Вектори, що вказують на точки на площині
# points_on_plane = np.stack([x, y, np.zeros_like(x)], axis=-1)
#
# # Вектори, що вказують на світлове джерело від кожної точки на площині
# light_directions = light_position - points_on_plane
#
# # Вектори, що вказують на спостерігача від кожної точки на площині
# observer_directions = observer_position - points_on_plane
#
# normals = normals.astype('float64')
# light_directions = light_directions.astype('float64')
# observer_directions = observer_directions.astype('float64')
#
#
# # Нормалізація векторів
# normals /= np.linalg.norm(normals)
# light_directions /= np.linalg.norm(light_directions, axis=-1, keepdims=True)
# observer_directions /= np.linalg.norm(observer_directions, axis=-1, keepdims=True)
#
# # Розрахунок освітленості (косинус кута між нормаллю, вектором до джерела та вектором до спостерігача)
# brightness = np.maximum(np.sum(normals * (light_directions + observer_directions), axis=-1), 0)
#
# # Відображення результатів в 3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(x, y, brightness, cmap='viridis', rstride=5, cstride=5, alpha=0.8)
#
# ax.scatter(light_position[0], light_position[1], light_position[2], color='red', marker='o', label='Light Source')
# ax.scatter(observer_position[0], observer_position[1], observer_position[2], color='blue', marker='o', label='Observer')
#
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Brightness')
# ax.set_title('Lighting on a Plane')
# ax.legend()
#
# plt.show()

8.

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# def generate_plane_surface(size=10):
#     x = np.linspace(0, size, 100)
#     y = np.linspace(0, size, 100)
#     x, y = np.meshgrid(x, y)
#     z = np.zeros_like(x)
#     return x, y, z
#
# def generate_light_source_position(size=10):
#     return size / 2, size / 2, 10
#
# def compute_shadow_coordinates(x, y, z, light_source):
#     shadow_coordinates = np.zeros_like(x)
#     for i in range(x.shape[0]):
#         for j in range(x.shape[1]):
#             point = np.array([x[i, j], y[i, j], z[i, j]])
#             light_vector = np.array(light_source) - point
#             shadow_coordinates[i, j] = np.dot(light_vector, [0, 0, -1])  # Projection on the z-axis
#
#     return shadow_coordinates
#
# def plot_3d_surface_with_shadow(x, y, z, shadow_coordinates):
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
#
#     ax.plot_surface(x, y, z, color='b', alpha=0.6, rstride=100, cstride=100)
#     ax.plot_surface(x, y, shadow_coordinates, color='gray', alpha=0.3, rstride=100, cstride=100)
#
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title('3D Surface with Shadow')
#
#     plt.show()
#
# # Генеруємо площину
# plane_surface = generate_plane_surface()
#
# # Генеруємо позицію джерела світла
# light_source_position = generate_light_source_position()
#
# # Обчислюємо координати тіні
# shadow_coordinates = compute_shadow_coordinates(*plane_surface, light_source_position)
#
# # Візуалізуємо площину та тінь
# plot_3d_surface_with_shadow(*plane_surface, shadow_coordinates)

9.

