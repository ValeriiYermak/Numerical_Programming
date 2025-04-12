1.
# import numpy as np
# import sympy as sp
# import matplotlib.pyplot as plt
#
# def my_taylor_series(func, x0, n_terms):
#     """
#     Розклад у ряд Тейлора без використання вбудованих функцій.
#
#     :param func: Функція, яку будемо розкладати.
#     :param x0: Точка розкладання.
#     :param n_terms: Кількість членів ряду Тейлора.
#     :return: Ряд Тейлора
#     """
#     taylor_series = 0
#     for n in range(n_terms):
#         # Обчислення n-го члена ряду Тейлора
#         term = func.diff(x, n).subs(x, x0) / sp.factorial(n) * (x - x0)**n
#         taylor_series += term
#     return taylor_series
#
# # Визначення символьної змінної та функції
# x = sp.symbols('x')
# func = sp.sin(x) + 6*sp.cos(x)+- x**2
#
# # Точка розкладання та кількість членів ряду Тейлора
# x0 = 0
# n_terms = 15
#
# # Розклад у ряд Тейлора
# taylor_series = my_taylor_series(func, x0, n_terms)
#
# # Компіляція функцій для NumPy
# func_np = sp.lambdify(x, func, 'numpy')
# taylor_np = sp.lambdify(x, taylor_series, 'numpy')
#
# # Генерація значень для графіків
# x_vals = np.linspace(-2 * np.pi, 2 * np.pi, 100)
# y_vals_func = func_np(x_vals)
# y_vals_taylor = taylor_np(x_vals)
#
# # Побудова графіка
# plt.plot(x_vals, y_vals_func, label='Функція')
# plt.plot(x_vals, y_vals_taylor, label=f'Ряд Тейлора (до {n_terms}-го члена)')
# plt.title(f'Розклад у ряд Тейлора для sin(x) в точці x={x0}')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.grid(True)
# plt.show()

2.

# import sympy as sp
#
# # Визначення символьної змінної
# x = sp.symbols('x')
#
# # Визначення функції експоненти та її розклад у ряд Тейлора
# exp_func = sp.exp(-x**2)
# taylor_series_exp = sp.series(exp_func, x, 0, 15).removeO()  # Ряд Тейлора до 5-го члена
#
# # Обчислення інтеграла за допомогою ряду Тейлора
# integral_taylor_exp = sp.integrate(taylor_series_exp, (x, 0, 1))
#
# # Вивід результатів
# print("Функція експоненти:", exp_func)
# print("Ряд Тейлора для експоненти:", taylor_series_exp)
# print("Інтеграл за допомогою ряду Тейлора:", integral_taylor_exp.evalf())

3.

# import numpy as np
# import matplotlib.pyplot as plt
#
# # Signal parameters
# amplitude = 1.0  # Amplitude of the sinusoidal signal
# frequency1 = 1  # Frequency of the sinusoidal signal (in Hertz)
# frequency2 = 3/2  # Frequency of the sinusoidal signal (in Hertz)
# phase = 0   # Initial phase of the sinusoidal signal (in radians)
#
# # Generating the time vector
# time = np.linspace(0, 2 * np.pi, 1000)
#
# # Generating the sinusoidal signal
# signal = time**2 + amplitude * np.sin(2 * np.pi * frequency1 * time + phase) + amplitude * np.cos(2 * np.pi * frequency2 * time + phase)
#
# # Plotting the graph
# plt.plot(time, signal, label='Sinusoidal Signal')
# plt.title('Sinusoidal Signal')
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.legend()
# plt.grid(True)
# plt.show()
4.
#
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import square, sawtooth
#
#
# # Налаштування параметрів сигналів
# t = np.linspace(0, 1, 1000, endpoint=False)  # 1 секунда, 1000 точок
# frequency = 5  # Частота сигналів, Гц
#
#
# # Словник сигналів для автоматичної генерації
# signals = {
#     "Синусоїдальний": np.sin(2 * np.pi * frequency * t),
#     "Косинусоїдальний": np.cos(2 * np.pi * frequency * t),
#     "Прямокутний": square(2 * np.pi * frequency * t),
#     "Трикутний": sawtooth(2 * np.pi * frequency * t, width=0.5),
#     "Пилообразний": sawtooth(2 * np.pi * frequency * t),
#     "Шумовий": np.random.normal(0, 0.5, len(t))  # Білий шум
# }
#
# # Візуалізація сигналів
# plt.figure(figsize=(12, 10))
# for i, (title, signal) in enumerate(signals.items(), start=1):
#     plt.subplot(len(signals), 1, i)
#     plt.plot(t, signal, lw=1.5)
#     plt.title(title, fontsize=10)
#     plt.xlabel("Час (с)", fontsize=9)
#     plt.ylabel("Амплітуда", fontsize=9)
#     plt.grid(True)
#
# plt.tight_layout()
# plt.show()
5.

# import numpy as np
# import matplotlib.pyplot as plt
#
# # Signal parameters
# amplitude = 1.0  # Amplitude of the triangular signal
# frequency = 1.0   # Frequency of the triangular signal (in Hertz)
# phase = 0.0      # Initial phase of the triangular signal (in radians)
#
# # Generating the time vector
# time = np.linspace(0, 2 * np.pi, 1000)
#
# # Generating the triangular signal
# signal = amplitude * np.abs(2 * (frequency * time / (2 * np.pi) + phase / (2 * np.pi) - 0.25) % 1 - 0.5) * 2
#
# # Plotting the graph
# plt.plot(time, signal, label='Triangular Signal')
# plt.title('Triangular Signal')
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.ylim(-1.5, 1.5)  # Setting limits for better visualization of the triangular signal
# plt.legend()
# plt.grid(True)
# plt.show()

6.

# import numpy as np
# import matplotlib.pyplot as plt
#
# # Задаємо діапазон значень x
# x = np.linspace(-np.pi, np.pi, 1000)
#
# # Функції в системі
# f1 = np.ones(np.shape(x)[0])
# f2 = np.sin(x)
# f3 = np.sin(2 * x)
# f4 = np.sin(3 * x)
# f5 = np.sin(4 * x)
#
# # Побудова графіків на різних підграфіках
# plt.figure(figsize=(10, 8))
#
# # Графік для функції 1
# plt.subplot(3, 2, 1)
# plt.plot(x, f1)
# plt.title('1')
#
# # Графік для sin(x)
# plt.subplot(3, 2, 2)
# plt.plot(x, f2)
# plt.title('sin(x)')
#
# # Графік для sin(2x)
# plt.subplot(3, 2, 3)
# plt.plot(x, f3)
# plt.title('sin(2x)')
#
# # Графік для sin(3x)
# plt.subplot(3, 2, 4)
# plt.plot(x, f4)
# plt.title('sin(3x)')
#
# # Графік для sin(4x)
# plt.subplot(3, 2, 5)
# plt.plot(x, f5)
# plt.title('sin(4x)')
#
# # Задаємо відступи між підграфіками
# plt.tight_layout()
#
# plt.show()

7.
# import sympy as sp
# from sympy import symbols, integrate
#
# # Визначте символи та функцію, яку потрібно інтегрувати
# x, n = symbols('x n')
# f = sp.sin(n*x)
#
# # Визначте межі інтегрування
# a = -sp.pi
# b = sp.pi
#
# # Обчисліть визначений інтеграл
# result = 1/(2*sp.pi) * integrate(f, (x, a, b))
#
# # Виведіть результат
# print(f"Визначений інтеграл від {a} до {b} для функції {f}: {result}")

8.
# import numpy as np
# import sympy as sp
# import matplotlib.pyplot as plt
#
# # Задаємо символьну змінну x
# x = sp.symbols('x')
#
# # Signal parameters
# amplitude = 1.0  # Amplitude of the sinusoidal signal
# frequency1 = 1  # Frequency of the sinusoidal signal (in Hertz)
# frequency2 = 3/2  # Frequency of the sinusoidal signal (in Hertz)
# phase = 0   # Initial phase of the sinusoidal signal (in radians)
# T = 2
# omega = (2 * sp.pi) / T
#
# # Задаємо функцію, яку будемо розкладати
# func =  amplitude * sp.sin(2 * sp.pi * frequency1 * x + phase) + amplitude * sp.cos(2 * sp.pi * frequency2 * x + phase)
# # func = x**2
#
# # Функція для обчислення дійсних коефіцієнтів ряду Фур'є
# def calculate_fourier_coefficients(func, n_terms=10):
#     # a_0 = (2 / sp.pi) * sp.integrate(func, (x, 0, sp.pi))
#     # a_n = [(2 / sp.pi) * sp.integrate(func * sp.cos(i * x), (x, 0, sp.pi)) for i in range(1, n_terms + 1)]
#     # b_n = [(2 / sp.pi) * sp.integrate(func * sp.sin(i * x), (x, 0, sp.pi)) for i in range(1, n_terms + 1)]
#
#     a_0 = (2 / T) * sp.integrate(func, (x, -T/2, T/2))
#     a_n = [(2 / T) * sp.integrate(func * sp.cos(omega * i * x), (x, -T/2, T/2)) for i in range(1, n_terms + 1)]
#     b_n = [(2 / T) * sp.integrate(func * sp.sin(omega * i * x), (x, -T/2, T/2)) for i in range(1, n_terms + 1)]
#
#     return a_0, a_n, b_n
#
# # Функція для обчислення суми ряду Фур'є
# def fourier_series(x_val, a_0, a_n, b_n, n_terms, omega):
#     series_sum = a_0 / 2
#     for i in range(1, n_terms + 1):
#         term = a_n[i - 1] * sp.cos(omega * i * x_val) + b_n[i - 1] * sp.sin(omega * i * x_val)
#         series_sum += term
#     return series_sum
#
# n_terms = 10
#
# # Отримуємо дійсні коефіцієнти ряду Фур'є
# a_0, a_n, b_n = calculate_fourier_coefficients(func, n_terms=n_terms)
#
# # Отримуємо значення x для побудови графіка
# x_values = np.linspace(0, 2*np.pi, 1000)
#
# # Отримуємо значення функції та ряду Фур'є
# f_values = [func.evalf(subs={x: val}) for val in x_values]
# fourier_values = [fourier_series(x_val, a_0, a_n, b_n, n_terms, omega).evalf() for x_val in x_values]
#
# # Побудова графіків
# plt.plot(x_values, f_values, label='Original Function')
# plt.plot(x_values, fourier_values, label='Fourier Series (10 terms)')
# plt.legend()
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.title('Fourier Series Approximation')
# plt.show()
9.

# import matplotlib.pyplot as plt
# import numpy as np
#
# # Функція для візуалізації комплексного числа
# def plot_complex_number(z, color='blue', label=None):
#     plt.scatter(z.real, z.imag, color=color, label=label)
#     plt.plot([0, z.real], [0, z.imag], color=color, linestyle='--')
#     plt.annotate(f'({z.real:.2f}, {z.imag:.2f})', (z.real, z.imag), textcoords="offset points", xytext=(-15,-10), ha='center')
#
# # Задаємо декілька комплексних чисел
# z1 = 2 + 3j
# z2 = -1 - 1j
# z3 = 4 * np.exp(1j * np.pi/4)  # Використовуємо формулу Ейлера для створення комплексного числа
#
# # Відображаємо комплексні числа у комплексній площині
# plt.figure(figsize=(8, 8))
# plot_complex_number(z1, color='red', label='z1')
# plot_complex_number(z2, color='green', label='z2')
# plot_complex_number(z3, color='blue', label='z3')
#
# # Налаштування графіка
# plt.axhline(0, color='black',linewidth=0.5)
# plt.axvline(0, color='black',linewidth=0.5)
# plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
# plt.title('Візуалізація комплексних чисел у комплексній площині')
# plt.xlabel('Дійсна вісь')
# plt.ylabel('Уявна вісь')
# plt.legend()
# plt.show()

10.

# import matplotlib.pyplot as plt
# import numpy as np
#
# # Функція для візуалізації комплексного числа та його спряженого
# def plot_complex_and_conjugate(z, color='blue', label=None):
#     plt.scatter(z.real, z.imag, color=color, label=label)
#     plt.plot([0, z.real], [0, z.imag], color=color, linestyle='--')
#     plt.scatter(z.real, -z.imag, color=color, marker='x')  # візуалізація спряженого
#     plt.plot([0, z.real], [0, -z.imag], color=color, linestyle='--')
#
# # Задаємо комплексне число
# z = 3 + 2j
#
# # Відображаємо комплексне число та його спряжене
# plt.figure(figsize=(8, 8))
# plot_complex_and_conjugate(z, color='blue', label='z')
#
# # Налаштування графіка
# plt.axhline(0, color='black', linewidth=0.5)
# plt.axvline(0, color='black', linewidth=0.5)
# plt.grid(color='gray', linestyle='--', linewidth=0.5)
# plt.title('Візуалізація комплексного числа та його спряженого')
# plt.xlabel('Дійсна частина')
# plt.ylabel('Уявна частина')
# plt.legend()
# plt.show()

11.

# import numpy as np
# import matplotlib.pyplot as plt
#
#
# import sympy as sp
# import matplotlib.pyplot as plt
#
# # Задаємо символьну змінну x
# x = sp.symbols('x')
#
# # Signal parameters
# amplitude = 1.0  # Amplitude of the sinusoidal signal
# frequency1 = 1  # Frequency of the sinusoidal signal (in Hertz)
# frequency2 = 3/2  # Frequency of the sinusoidal signal (in Hertz)
# phase = 0   # Initial phase of the sinusoidal signal (in radians)
# T = 2
# omega = (2 * sp.pi) / T
#
# # Задаємо функцію, яку будемо розкладати
# func = amplitude * sp.sin(2 * sp.pi * frequency1 * x + phase) + amplitude * sp.cos(2 * sp.pi * frequency2 * x + phase)
#
# # Функція для обчислення дійсних коефіцієнтів ряду Фур'є
# def calculate_fourier_coefficients(func, n_terms=10):
#
#     c_n = [(1 / T) * sp.integrate(func * sp.exp(- sp.I * omega * i * x), (x, -T/2, T/2)) for i in range(-n_terms, n_terms+1)]
#
#     return c_n
#
# # Функція для обчислення суми ряду Фур'є
# def fourier_series(x_val, c_n, n_terms, omega):
#     series_sum = 0
#     for i in range(-n_terms, n_terms + 1):
#         term = c_n[i] * sp.exp(sp.I * omega * i * x_val)
#         series_sum += term
#
#     return series_sum
#
# n_terms = 5
#
# # Отримуємо дійсні коефіцієнти ряду Фур'є
# c_n = calculate_fourier_coefficients(func, n_terms=n_terms)
#
# # Отримуємо значення x для побудови графіка
#
# x_values = np.linspace(0, 2*np.pi, 100)
# # Отримуємо значення функції та ряду Фур'є
# f_values = [func.evalf(subs={x: val}) for val in x_values]
#
# fourier_values = [fourier_series(x_val, c_n, n_terms, omega).evalf() for x_val in x_values]
#
# real_fourier_values = [sp.re(fourier_val).evalf() for fourier_val in fourier_values]
# im_fourier_values = [sp.im(fourier_val).evalf() for fourier_val in fourier_values]
#
# # Візуалізація результату
# plt.plot(x_values, np.real(f_values), label='Оригінальна функція (дійсна частина)')
# # plt.plot(x_values, np.imag(f_values), label='Оригінальна функція (уявна частина)')
# plt.plot(x_values, real_fourier_values, label='Ряд Фур`є (дійсна частина)')
# # plt.plot(x_values, im_fourier_values, label='Ряд Фур`є (уявна частина)')
# plt.legend()
# plt.show()

12.

# from scipy.fft import fft, rfft
# from scipy.fft import fftfreq, rfftfreq
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Оголошення класу Signal
# class Signal:
#     def __init__(self, amplitude=1, frequency=10, duration=1, sampling_rate=100.0, phase=0):
#         self.amplitude = amplitude
#         self.frequency = frequency
#         self.duration = duration
#         self.sampling_rate = sampling_rate
#         self.phase = phase
#         self.time_step = 1.0 / self.sampling_rate
#         self.time_axis = np.arange(0, self.duration, self.time_step)
#
#     def sine(self):
#         return self.amplitude * np.sin(2 * np.pi * self.frequency * self.time_axis + self.phase)
#
#     def cosine(self):
#         return self.amplitude * np.cos(2 * np.pi * self.frequency * self.time_axis + self.phase)
#
#
# # **Оголошення об'єктів класу ЗА МЕЖАМИ класу**
# signal_1hz = Signal(amplitude=3, frequency=1, sampling_rate=200, duration=2)
# sine_1hz = signal_1hz.sine()
#
# signal_20hz = Signal(amplitude=1, frequency=20, sampling_rate=200, duration=2)
# sine_20hz = signal_20hz.sine()
#
# signal_10hz = Signal(amplitude=0.5, frequency=10, sampling_rate=200, duration=2)
# sine_10hz = signal_10hz.sine()
#
# # Сума трьох сигналів
# signal = sine_1hz + sine_20hz + sine_10hz
#
# # Побудова графіка
# plt.plot(signal_1hz.time_axis, signal, 'b')
# plt.xlabel('Time [sec]')
# plt.ylabel('Amplitude')
# plt.title('Sum of three signals')
# plt.show()


13.
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.fft import fft, fftfreq
# from scipy.fft import rfft, rfftfreq
#
# # Створення тестового сигналу
# sampling_rate = 200.0  # Частота дискретизації
# duration = 2.0  # Тривалість сигналу (секунди)
# N = int(sampling_rate * duration)  # Загальна кількість точок
#
# time_axis = np.linspace(0, duration, N, endpoint=False)  # Часова вісь
# signal = 3 * np.sin(2 * np.pi * 1 * time_axis) + 1 * np.sin(2 * np.pi * 20 * time_axis)  # Сигнал
#
# # Обчислення перетворення Фур'є
# fourier = fft(signal)
# frequency_axis = fftfreq(N, d=1.0 / sampling_rate)
# norm_amplitude = np.abs(fourier) / N  # Нормалізація амплітуди
#
# # Побудова графіка
# plt.plot(frequency_axis[:N // 2], norm_amplitude[:N // 2])  # Відображаємо лише додатні частоти
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('Amplitude')
# plt.title('Spectrum')
# plt.show()
#
#
# # Building a class Fourier for better use of Fourier Analysis.
# class Fourier:
#     """
#     Apply the Discrete Fourier Transform (DFT) on the signal using the Fast Fourier
#     Transform (FFT) from the scipy package.
#
#     Example:
#       fourier = Fourier(signal, sampling_rate=2000.0)
#     """
#
#     def __init__(self, signal, sampling_rate):
#         """
#         Initialize the Fourier class.
#
#         Args:
#             signal (np.ndarray): The samples of the signal
#             sampling_rate (float): The sampling per second of the signal
#
#         Additional parameters,which are required to generate Fourier calculations, are
#         calculated and defined to be initialized here too:
#             time_step (float): 1.0/sampling_rate
#             time_axis (np.ndarray): Generate the time axis from the duration and
#                                   the time_step of the signal. The time axis is
#                                   for better representation of the signal.
#             duration (float): The duration of the signal in seconds.
#             frequencies (numpy.ndarray): The frequency axis to generate the spectrum.
#             fourier (numpy.ndarray): The DFT using rfft from the scipy package.
#         """
#         self.signal = signal
#         self.sampling_rate = sampling_rate
#         self.time_step = 1.0 / self.sampling_rate
#         self.duration = len(self.signal) / self.sampling_rate
#         self.time_axis = np.arange(0, self.duration, self.time_step)
#         self.frequencies = rfftfreq(len(self.signal), d=self.time_step)
#         self.fourier = rfft(self.signal)
#
#     # Generate the actual amplitudes of the spectrum
#     def amplitude(self):
#         """
#         Method of Fourier
#
#         Returns:
#             numpy.ndarray of the actual amplitudes of the sinusoids.
#         """
#         return 2 * np.abs(self.fourier) / len(self.signal)
#
#     # Generate the phase information from the output of rfft
#     def phase(self, degree=False):
#         """
#         Method of Fourier
#
#         Args:
#             degree: To choose the type of phase representation (Radian, Degree).
#                     By default, it's in radian.
#
#         Returns:
#             numpy.ndarray of the phase information of the Fourier output.
#         """
#         return np.angle(self.fourier, deg=degree)
#
#     # Plot the spectrum
#     def plot_spectrum(self, interactive=False):
#         """
#         Plot the Spectrum (Frequency Domain) of the signal either using the matplotlib
#         package, or plot it interactive using the plotly package.
#
#         Args:
#             interactive: To choose if you want the plot interactive (True), or not
#             (False). The default is the spectrum non-interactive.
#
#         Retruns:
#             A plot of the spectrum.
#         """
#         # When the argument interactive is set to True:
#         if interactive:
#             self.trace = go.Line(x=self.frequencies, y=self.amplitude())
#             self.data = [self.trace]
#             self.layout = go.Layout(title=dict(text='Spectrum',
#                                                x=0.5,
#                                                xanchor='center',
#                                                yanchor='top',
#                                                font=dict(size=25, family='Arial, bold')),
#                                     xaxis=dict(title='Frequency[Hz]'),
#                                     yaxis=dict(title='Amplitude'))
#         self.fig = go.Figure(data=self.data, layout=self.layout)
#         return self.fig.show()
#
#     # When the argument interactive is set to False:
#     else:
#     plt.figure(figsize=(10, 6))
#     plt.plot(self.frequencies, self.amplitude())
#     plt.title('Spectrum')
#     plt.ylabel('Amplitude')
#     plt.xlabel('Frequency[Hz]')
#
#
# # Plot the Signal and the Spectrum interactively
# def plot_time_frequency(self, t_ylabel="Amplitude", f_ylabel="Amplitude",
#                         t_title="Signal (Time Domain)",
#                         f_title="Spectrum (Frequency Domain)"):
#     """
#     Plot the Signal in Time Domain and Frequency Domain using plotly.
#
#     Args:
#         t_ylabel (String): Label of the y-axis in Time-Domain
#         f_ylabel (String): Label of the y-axis in Frequency-Domain
#         t_title (String): Title of the Time-Domain plot
#         f_title (String): Title of the Frequency-Domain plot
#
#     Returns:
#         Two figures: the first is the time-domain, and the second is the
#                      frequency-domain.
#     """
#     # The Signal (Time-Domain)
#     self.time_trace = go.Line(x=self.time_axis, y=self.signal)
#     self.time_domain = [self.time_trace]
#     self.layout = go.Layout(title=dict(text=t_title,
#                                        x=0.5,
#                                        xanchor='center',
#                                        yanchor='top',
#                                        font=dict(size=25, family='Arial, bold')),
#                             xaxis=dict(title='Time[sec]'),
#                             yaxis=dict(title=t_ylabel),
#                             width=1000,
#                             height=400)
#     fig = go.Figure(data=self.time_domain, layout=self.layout)
#     fig.show()
#     # The Spectrum (Frequency-Domain)
#     self.freq_trace = go.Line(x=self.frequencies, y=self.amplitude())
#     self.frequency_domain = [self.freq_trace]
#     self.layout = go.Layout(title=dict(text=f_title,
#                                        x=0.5,
#                                        xanchor='center',
#                                        yanchor='top',
#                                        font=dict(size=25, family='Arial, bold')),
#                             xaxis=dict(title='Frequency[Hz]'),
#                             yaxis=dict(title=f_ylabel),
#                             width=1000,
#                             height=400)
#     fig = go.Figure(data=self.frequency_domain, layout=self.layout)
#     fig.show()


14.

# import os
# import librosa
# import zipfile
# from urllib.request import urlretrieve
# from IPython.display import Audio
#
# # Завантаження та розархівування датасету
# url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
# zip_file_path = "ESC-50-master.zip"
# download_path = "./ESC-50-master/"
# if not os.path.exists(download_path):
#     urlretrieve(url, zip_file_path)
#     with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#         zip_ref.extractall(download_path)
#     os.remove(zip_file_path)

15.
# import librosa.display
# import matplotlib.pyplot as plt
#
# # Зображення графіка амплітуди від часу
# plt.figure(figsize=(14, 5))
# librosa.display.waveshow(y, sr=sr)
# plt.title('Графік амплітуди від часу')
# plt.xlabel('Час (с)')
# plt.ylabel('Амплітуда')
# plt.show()

16.

# import numpy as np
# from scipy.fft import fft
#
# mystery_signal = y
# num_samples = sr
#
# # Perform the Fourier Transform on the mystery signal
# mystery_signal_fft = fft(mystery_signal)
#
# # Compute the amplitude spectrum
# amplitude_spectrum = np.abs(mystery_signal_fft)
#
# # Normalize the amplitude spectrum
# amplitude_spectrum = amplitude_spectrum / np.max(amplitude_spectrum)
#
# # Compute the frequency array
# freqs = np.fft.fftfreq(num_samples, 1 / sampling_rate)
#
# # Plot the amplitude spectrum in the frequency domain
# plt.plot(freqs[:num_samples // 2], amplitude_spectrum[:num_samples // 2])
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Normalized Amplitude")
# plt.title("Amplitude Spectrum of the Mystery Signal")
# plt.show()
#
# # Find the dominant frequencies
# threshold = 0.6
# dominant_freq_indices = np.where(amplitude_spectrum[:num_samples // 2] >= threshold)[0]
# dominant_freqs = freqs[dominant_freq_indices]
#
# print("Dominant Frequencies: ", dominant_freqs)

17.
#
# import librosa
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Завантаження аудіофайлу
# filename = "path/to/your/audio/file.wav"  # Вкажи правильний шлях до файлу
# y, sr = librosa.load(filename, sr=None)  # Завантажуємо файл із його нативною частотою дискретизації
#
# def spectrogram(samples, sample_rate, stride_ms=10.0,
#                 window_ms=20.0, max_freq=None, eps=1e-14):
#
#     stride_size = int(0.001 * sample_rate * stride_ms)
#     window_size = int(0.001 * sample_rate * window_ms)
#
#     # Extract strided windows
#     truncate_size = (len(samples) - window_size) % stride_size
#     samples = samples[:len(samples) - truncate_size]
#     nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
#     nstrides = (samples.strides[0], samples.strides[0] * stride_size)
#     windows = np.lib.stride_tricks.as_strided(samples, shape=nshape, strides=nstrides)
#
#     # Window weighting, squared Fast Fourier Transform (fft), scaling
#     weighting = np.hanning(window_size)[:, None]
#     fft = np.fft.rfft(windows * weighting, axis=0)
#     fft = np.absolute(fft)**2
#
#     scale = np.sum(weighting**2) * sample_rate
#     fft[1:-1, :] *= (2.0 / scale)
#     fft[(0, -1), :] /= scale
#
#     # Logarithm of the spectrogram
#     specgram = np.log(fft + eps)
#     return specgram
#
# # Отримання спектральної матриці
# spect_matrix_db = spectrogram(y, sr)
#
# # Функція для візуалізації спектрограми
# def spect_show(spect_matrix):
#     plt.imshow(spect_matrix, cmap='viridis', aspect='auto', origin='lower')
#     plt.colorbar(label='Інтенсивність (дБ)')
#     plt.title('Візуалізація спектральної матриці')
#     plt.xlabel('Номер фрейму')
#     plt.ylabel('Частота')
#     plt.show()
#
# # Відображення спектрограми
# spect_show(spect_matrix_db)
#
