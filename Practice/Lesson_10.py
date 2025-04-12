1.
# from scipy.stats import norm
#
# # Параметри розподілу
# mu = 0  # середнє значення
# sigma = 1  # стандартне відхилення
#
# # Задана точка, для якої ми хочемо обчислити ймовірність P(X < x)
# x = 2
#
# # Обчислення функції розподілу
# probability = norm.cdf(x, mu, sigma)
#
# print(f'P(X < {x}) = {probability:.4f}')

2.

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import norm
# import pandas as pd
#
# # Параметри розподілу
# mu = 0
# sigma = 1
#
# # Створення значень x для графіка
# x_values = np.linspace(-3, 3, 1000)
#
# # Обчислення значень функції щільності ймовірностей (PDF)
# pdf_values = norm.pdf(x_values, mu, sigma)
#
# # Побудова графіка
# plt.figure(figsize=(8, 6))
# plt.plot(x_values, pdf_values, label=r'$f(x|0, 1) = \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}}$', color='blue')
# plt.fill_between(x_values, pdf_values, color='skyblue', alpha=0.3)
#
# # Позначення точки x = 2
# plt.scatter([2], [norm.pdf(2, mu, sigma)], color='red', label='P(X < 2)', zorder=5)
# plt.annotate('P(X < 2)', xy=(2, norm.pdf(2, mu, sigma)), xytext=(2.5, 0.25),
#              arrowprops=dict(facecolor='black', arrowstyle='->'),
#              fontsize=10)
#
# # Налаштування графіка
# plt.title('Нормальний розподіл Гауса')
# plt.xlabel('x')
# plt.ylabel('Щільність ймовірностей')
# plt.legend()
# plt.grid(True)
#
# # Показати графік
# plt.show()
3.
# import pandas as pd
# import numpy as np
#
# data = {'Height': [140, 145, 135, 169, 165, 142, 168, 141, 159, 160, 172],
#         'Gender': ['F', 'F', 'F', 'M', 'F', 'M', 'M', 'F', 'F', 'M', 'M']}
#
# df = pd.DataFrame(data)
# print(df)
#
# def create_interval_dataframe(dataframe, step):
#     """
#     Перетворює датафрейм з ростом та статтю у інтервальний датафрейм.
#
#     Параметри:
#     - dataframe: pd.DataFrame, вихідний датафрейм зі стовпцями 'Зріст' та 'Стать'.
#     - step: int, крок для побудови інтервалів.
#
#     Повертає:
#     - pd.DataFrame, інтервальний датафрейм зі стовпцями 'Інтервал', 'Чоловіки', 'Жінки'.
#     """
#     # Розрахунок мінімального та максимального зросту
#     min_height = dataframe['Height'].min()
#     max_height = dataframe['Height'].max()
#
#     # Створення категорій для інтервалів
#     categories = pd.cut(dataframe['Height'], bins=np.arange(min_height, max_height + step, step), right=False)
#
#     # Додавання нового стовпця 'Інтервал'
#     dataframe['Interval'] = categories
#
#     # Сортування індексів
#     dataframe = dataframe.sort_values(by='Interval')
#
#     # Групування за інтервалами та статтю, підрахунок частот
#     grouped_df = dataframe.groupby(['Interval', 'Gender']).size().unstack().reset_index()
#
#     return grouped_df
#
# # Приклад використання:
# # Визначення кроку
# step = 5
#
# result_df = create_interval_dataframe(df, step)
#
# print(result_df)
# import pandas as pd
# import scipy.stats as stats
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# data_f = result_df[['Interval', 'F']]
#
# df_f = pd.DataFrame(data_f)
#
#
# # Побудова графіку інтервального розподілу частот
# plt.figure(figsize=(8, 6))
# plt.bar(range(len(df_f)), df_f['F'], tick_label=df_f['Interval'].astype(str))
# plt.title('Інтервальний розподіл частот')
# plt.xlabel('Інтервали')
# plt.ylabel('Частоти')
# plt.grid(axis='y', linestyle='--', alpha=0.7)
#
# # Побудова Q-Q графіку
# plt.figure(figsize=(8, 6))
# stats.probplot(df_f['F'], dist="norm", plot=plt)
# plt.title('Q-Q графік')
# plt.show()
#
# # Перевірка нормальності тестом Шапіро-Уілка
# stat, p_value = stats.shapiro(df_f['F'])
# print(f'Shapiro-Wilk test statistic: {stat}, p-value: {p_value}')
#
# # Визначення рівня значущості
# alpha = 0.05
#
# # Перевірка гіпотези про нормальність
# if p_value > alpha:
#     print("Розподіл є нормальним (не відхиляється від нормального)")
# else:
#     print("Розподіл не є нормальним (відхиляється від нормального)")

4.

# import pandas as pd
# import scipy.stats as stats
# import numpy as np
# import matplotlib.pyplot as plt
#
# df_m = result_df[['Interval', 'M']]
#
#
# # Побудова графіку інтервального розподілу частот
# plt.figure(figsize=(8, 6))
# plt.bar(range(len(df_m)), df_m['M'], tick_label=df_m['Interval'].astype(str))
# plt.title('Інтервальний розподіл частот')
# plt.xlabel('Interval')
# plt.ylabel('M')
# plt.grid(axis='y', linestyle='--', alpha=0.7)
#
# # Побудова Q-Q графіку
# plt.figure(figsize=(8, 6))
# stats.probplot(df_m['M'], dist="norm", plot=plt)
# plt.title('Q-Q графік')
# plt.show()
#
# # Перевірка нормальності тестом Шапіро-Уілка
# stat, p_value = stats.shapiro(df_m['M'])
# print(f'Shapiro-Wilk test statistic: {stat}, p-value: {p_value}')
#
# # Визначення рівня значущості
# alpha = 0.05
#
# # Перевірка гіпотези про нормальність
# if p_value > alpha:
#     print("Розподіл є нормальним (не відхиляється від нормального)")
# else:
#     print("Розподіл не є нормальним (відхиляється від нормального)")

5.

# import math
# std_m = df[df['Gender']=='M']['Height'].std()
# sigma_m = std_m**2
# mu_m = df[df['Gender']=='M']['Height'].mean()
#
#
# std_f = df[df['Gender']=='F']['Height'].std()
# sigma_f = std_f**2
# mu_f = df[df['Gender']=='F']['Height'].mean()
#
#
# def norm_distr(x, mu, sigma):
#     return 1 / (sigma * math.sqrt(2 * math.pi)) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)
#
#
# P_h_m = norm_distr(152, mu_m, sigma_m)
# P_h_m

6.
# import numpy as np
# x = [1, 2, 3, 4, 5]
# y = [2, 4, 6, 8, 11]
# z = [-1, -2, -3, -4, -5]
#
# #Обчислимо дисперсії кожного ряду даних
# var_x = np.var(x)*len(x)/(len(x)-1)
# print(var_x)
#
# var_y = np.var(y)*len(y)/(len(y)-1)
# print(var_y)
#
# var_z = np.var(z)*len(z)/(len(z)-1)
# print(var_z)
#
# X = np.stack((x, y, z), axis=0)
#
# print('Матриця коваріації\n',np.cov(X))

7.
# import pandas as pd
# import numpy as np
# data = {'Height': [140, 145, 135, 169, 165, 142, 168, 141, 159, 160, 172],
#         'Gender': ['F', 'F', 'F', 'M', 'F', 'M', 'M', 'F', 'F', 'M', 'M']}
#
# df = pd.DataFrame(data)
# print(df)
#
# from numpy.linalg import inv
# x_m = np.array(df[df['Gender']=='M']['Height'])
# print(x_m)
#
# x_f = np.array(df[df['Gender']=='F']['Height'])
# print(x_f)
#
# x_m_cov = np.var(x_m)*len(x)/(len(x)-1)
# print(x_m_cov)
#
# x_f_cov = np.var(x_f)*len(x)/(len(x)-1)
# print(x_f_cov)
#
# inv_cov_m = inv(np.array([[x_m_cov]]))
# print(inv_cov_m)
#
# inv_cov_f = inv(np.array([[x_f_cov]]))
# print(inv_cov_f)
#
# P_m = len(df[df['Gender']=='M'])/len(df)
# print(P_m)
#
# P_f = len(df[df['Gender']=='F'])/len(df)
# print(P_f)
#
# def g(x, X_mean, X_cov, X_inv_cov, X_prob):
#     g_i = -1/2*(x - X_mean)*(x - X_mean)*X_inv_cov - 1/2*np.log(X_cov)+np.log(X_prob)
#     return g_i
#
# g_m = g(152, np.mean(x_m), x_m_cov, inv_cov_m, P_m)
# print(g_m)
#
# g_f = g(152, np.mean(x_f), x_f_cov, inv_cov_f, P_f)
# print(g_f)
#
#
# values = np.array(list([g_m[0], g_f[0]])).transpose()
# print(values)
#
# likelihood = np.exp(values - values.max(axis=1)[:, np.newaxis])
# res = likelihood / likelihood.sum(axis=1)[:, np.newaxis]
# print(res)

8.
import numpy as np
import pandas as pd

data = {'Height': [140, 145, 135, 169, 165, 142, 168, 141, 159, 160, 172],
    'Gender': ['F', 'F', 'F', 'M', 'F', 'M', 'M', 'F', 'F', 'M', 'M']}

df = pd.DataFrame(data)
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

height = np.array(df['Height'])
gender = np.array(df['Gender'])

qda = QuadraticDiscriminantAnalysis()
qda.fit(height.reshape(-1,1), gender)
x = np.array([152])

print('Значення дискримінантної функції\n', qda._decision_function(x.reshape(-1,1)))

print('Значення імовірностей класів\n', qda.predict_proba(x.reshape(-1,1)))