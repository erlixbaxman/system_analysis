import numpy as np
import matplotlib.pyplot as plt

# Создаем массив значений x от 0 до 10 с шагом 0.1
x = np.arange(0, 50, 0.1)

# Создаем массив значений y, используя функцию синуса
y = np.sin(x)

# Добавляем шум в массив значений y с помощью функции randn из библиотеки NumPy
y_noisy = y + 0.155 * np.random.randn(len(x))

# Создаем график синусоиды и шума с помощью библиотеки Matplotlib
plt.grid()
plt.plot(x, y, 'b', label='Синусоида')
plt.plot(x, y_noisy, 'r', label='Синусоида с шумом')
plt.legend()
plt.show()


# Объявляем функцию ядерной регрессии
def get_kernel_regression_estimation(y, h):
    sse = 10000
    approximated_y = []
    while sse > 1000:
        for i in range(len(y)):
            w = [(1 / np.sqrt(2 * np.pi)) * np.exp((-1 / 2) * (((i + 1) - j) / h) ** 2) for j in range(1, len(y))]
            approximated_y.append(sum(list(map(lambda w_value, y_value: w_value * y_value, w, y))) / sum(w))
        sse = sum([(y[i] - approximated_y[i]) ** 2 for i in range(len(y))])
        y = approximated_y[:]
        approximated_y = []
    return y


# Объявляем функцию построения
def plotting(x, y, h,results, title):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.grid()
    ax.plot(x, y_noisy)
    ax.plot(x, results[0])
    ax.plot(x, results[1])
    ax.plot(x, results[2])
    ax.legend(('y(x) с шумом',
               f'h1 = {h[0]}',
               f'h2 = {h[1]}',
               f'h3 = {h[2]}'))
    plt.show()


# Финальные штрихи
approximation_results = []
h = [1.5,10,20]
for i in h:
    approximation_results.append(get_kernel_regression_estimation(y_noisy,i))

plotting(x,y,h,approximation_results, 'Гауссовский процесс')