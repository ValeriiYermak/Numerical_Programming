# import numpy as np
# import matplotlib.pyplot as plt
#
# # Квадратична функція
# def quadratic_function(x):
#     return x**2 + 5*x + 6
#
# # Градієнт квадратичної функції
# def gradient(x):
#     return 2*x + 5
#
# # Алгоритм градієнтного спуску
# def gradient_descent(learning_rate, epochs, initial_x):
#     x_values = [initial_x]
#     for epoch in range(epochs):
#         current_x = x_values[-1]
#         grad = gradient(current_x)
#         new_x = current_x - learning_rate * grad
#         x_values.append(new_x)
#     return x_values
#
# # Параметри градієнтного спуску
# learning_rate = 0.1
# epochs = 20
# initial_x = 0
#
# # Запуск градієнтного спуску
# optimized_x_values = gradient_descent(learning_rate, epochs, initial_x)
#
# # Графік функції та градієнтного спуску
# x_values = np.linspace(-5, 2, 100)
# y_values = quadratic_function(x_values)
#
# plt.plot(x_values, y_values, label='f(x) = x^2 + 5x + 6')
# plt.scatter(optimized_x_values, [quadratic_function(x) for x in optimized_x_values], color='red', label='Градієнтний спуск')
# plt.title('Градієнтний спуск для квадратичної функції')
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.legend()
# plt.grid(True)
# plt.show()

2.
# def y_hat(w, x_val):
#     return w[1]*x_val + w[0]
#
# def de(x, y, w, ind):
#     m = len(x)
#     error = [y_hat(w, x[i]) - y[i] for i in range(len(x))]
#     if ind == 0:
#         res = 1/m*(sum(error))
#     if ind == 1:
#         res = 1/m*(sum([x[i]*error[i] for i in range(len(x))]))
#     return res
#
# def gd2(x, y, w_0, iterations, gamma):
#     w_i = w_0
#     for i in range(iterations):
#         w_p = w_i
#         w_i = [w_p[0] - gamma*de(x, y, w_p, 0), w_p[1] - gamma*de(x, y, w_p, 1)]
#         print('Iteration {} optimal w {}'.format(i, w_i))
#     return w_i
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Згенеруємо деякі випадкові дані
# np.random.seed(42)
# X = 2 * np.random.rand(100, 1)
# y = 4 + 3 * X + np.random.randn(100, 1)
#
# # Ініціалізуємо параметри моделі
# a, b = np.random.randn(), np.random.randn()
#
# # Градієнтний спуск
# learning_rate = 0.01
# n_iterations = 1000
#
# gd2(X, y, [a, b], n_iterations, learning_rate)

3.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# Step 1: Generate a synthetic dataset with moderate noise (same)
X, y = make_regression(n_samples=100, n_features=1, noise=5)
y = y.reshape(-1, 1)


# Step 2: Split the dataset into training and test sets (same)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Step 3: Define a more complex neural network model (same)
model = Sequential([
    Dense(1024, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(1024, activation='relu'),
    Dense(1024, activation='relu'),
    Dense(512, activation='relu'),
    Dense(1)
])


# Compile the model (same)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mse'])


# Step 4: Train the model for many epochs (too many)
history = model.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test), verbose=0)


# Combine history from both training phases (same)
history_combined_loss = history.history['loss']
history_combined_val_loss = history.history['val_loss']


# Step 5: Visualize the loss on both the training and test datasets
plt.figure(figsize=(12, 10))  # Adjust for readability


# Plot loss for overfitting
plt.plot(range(1, len(history_combined_loss) + 1), history_combined_loss, label='Train Loss')
plt.plot(range(1, len(history_combined_val_loss) + 1), history_combined_val_loss, label='Test Loss')
plt.title('Overfitting (Too Many Epochs)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

4.

# Step 3: Define a very simple neural network model (underfitting)
model = Sequential([
  Dense(1, activation='relu', input_shape=(X_train.shape[1],))  # Single neuron output layer
])


# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mse'])


# Step 4: Train the model with a sufficient number of epochs
history = model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), verbose=0)


# Combine history from both training phases (similar to overfitting example)
history_combined_loss = history.history['loss']
history_combined_val_loss = history.history['val_loss']


# Step 6: Visualize the loss on both the training and test datasets
plt.figure(figsize=(12, 10))  # Adjust for readability


# Plot loss for underfitting
plt.plot(range(1, 201), history_combined_loss, label='Train Loss')
plt.plot(range(1, 201), history_combined_val_loss, label='Test Loss')
plt.title('Underfitting (Simple Model)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

5.

from tensorflow.keras.callbacks import EarlyStopping

# Step 3: Define a more complex neural network model (same as overfitting)
model = Sequential([
    Dense(1024, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(1024, activation='relu'),
    Dense(1024, activation='relu'),
    Dense(512, activation='relu'),
    Dense(1)
])


# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mse'])


# Step 4: Define Early Stopping callback (introduce patience)
early_stopping = EarlyStopping(monitor='val_loss', patience=5)  # Monitor validation loss, stop after 5 epochs of no improvement


# Step 5: Train the model with Early Stopping
history = model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=0)


# Combine history from both training phases (similar to previous examples)
history_combined_loss = history.history['loss']
history_combined_val_loss = history.history['val_loss']


# Step 6: Visualize the loss on both the training and test datasets
plt.figure(figsize=(12, 10))  # Adjust for readability


# Plot loss for early stopping
plt.plot(range(1, len(history_combined_loss) + 1), history_combined_loss, label='Train Loss')
plt.plot(range(1, len(history_combined_val_loss) + 1), history_combined_val_loss, label='Test Loss')
plt.title('Early Stopping (Overfitting Prevention)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

6.

import sympy
x1 = sympy.symbols('x1')
x2 = sympy.symbols('x2')
x3 = sympy.symbols('x3')

def f(x1, x2, x3):
    return 3*(x1**2 + x2*x3)

u = f(x1, x2, x3)
print('f(x1, x2, x3) = ',u)

gradient_fun = [u.diff(x1), u.diff(x2), u.diff(x3)]
print('gradient function =', gradient_fun)

gradient_val = [g.subs({x1: 2, x2: 3, x3: 4}) for g in gradient_fun]
print('gradient values =', gradient_val)