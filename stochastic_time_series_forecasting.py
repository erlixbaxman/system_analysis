import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings('ignore')


def moving_average_method(real_x, real_y, test_x, test_y, data):
    mov_av_model = data.copy()
    ma_window = 5
    mov_av_model['sma_forecast'] = data.iloc[:, 1].rolling(ma_window).mean()
    mov_av_model['sma_forecast'][len(y):] = mov_av_model['sma_forecast'][len(real_y) - 1]

    rmse_test = np.sqrt(mean_squared_error(test_y, mov_av_model['sma_forecast'][len(real_y)-1:]))
    print(f'RMSE для скользящего среднего = {rmse_test}')

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(x, y, linewidth=1, label='real')
    ax.plot(test_x, test_y, 'r', linewidth=1, label='real last 14 days')
    ax.plot(mov_av_model['sma_forecast'][:len(real_y)], linewidth=1, label='moving average')
    ax.plot(mov_av_model['sma_forecast'][len(real_y) - 1:], 'g', linewidth=1, label='prediction')
    ax.set_xlabel('Days')
    ax.set_ylabel('Price, USD')
    ax.set_title('Moving Average')
    plt.grid()
    plt.legend()
    plt.show()


def exponential_smoothing_method(real_x, real_y, test_x, test_y):
    exp_model = ExponentialSmoothing(y, trend='add', seasonal='mul', seasonal_periods=15)
    exp_fit = exp_model.fit()
    result = exp_fit.predict(start=real_x[1], end=len(real_y) + len(test_y) - 1)

    rmse_test = np.sqrt(mean_squared_error(test_y, result[len(real_x) - 1:]))
    print(f'RMSE для экспоненциального сглаживания = {rmse_test}')

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(real_x, real_y, linewidth=1, label='real')
    ax.plot(test_x, test_y, 'r', linewidth=1, label='real last 14 days')
    ax.plot(real_x, result[:len(real_y)], linewidth=1, label='smoothing')
    ax.plot(test_x, result[len(real_y) - 1:], 'g', linewidth=1, label='prediction')
    ax.set_xlabel('Days')
    ax.set_ylabel('Price, USD')
    ax.set_title('Exponential Smoothing')
    plt.legend()
    plt.grid()
    plt.show()


def trend_method(real_x, real_y, test_x, test_y):
    polynomial_features = PolynomialFeatures(degree=3)

    train_poly = polynomial_features.fit_transform(x.reshape(-1, 1))
    test_poly = polynomial_features.fit_transform(test_x.reshape(-1, 1))

    model_train = LinearRegression()
    train_fit = model_train.fit(train_poly, y)

    model_test = LinearRegression()
    test_fit = model_test.fit(test_poly, test_y)

    train_pred = train_fit.predict(train_poly)
    test_pred = test_fit.predict(test_poly)

    rmse_test = np.sqrt(mean_squared_error(test_y, test_pred))
    r2_test = r2_score(test_y, test_pred)

    print(f'RMSE для полиномиальной регрессии = {rmse_test}')
    print(f'R2 для полиномиальной регрессии = {r2_test}\n')

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(real_x, real_y, linewidth=1, label='real')
    ax.plot(test_x, test_y, 'r', linewidth=1, label='real last 14 days')
    ax.plot(x[:len(real_x)+1], train_pred[:len(real_y)+1], linewidth=1, label='trend')
    ax.plot(x[len(real_x):], train_pred[len(real_y):], 'g', linewidth=1, label='prediction')
    ax.set_xlabel('Days')
    ax.set_ylabel('Price, USD')
    ax.set_title('Polynomial Trend')
    plt.grid()
    plt.legend()
    plt.show()


def ar_method(real_x, real_y, test_x, test_y):
    ar_model = ARIMA(real_y, order=(7, 0, 0))
    ma_fit = ar_model.fit()
    result = ma_fit.predict(start=real_x[1], end=len(real_y) + len(test_y) - 1)

    rmse_test = np.sqrt(mean_squared_error(test_y, result[len(real_x) - 1:]))
    print(f'RMSE для модели AR = {rmse_test}')

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(real_x, real_y, linewidth=1, label='real')
    ax.plot(test_x, test_y, 'r', linewidth=1, label='real last 14 days')
    ax.plot(real_x, result[:len(real_y)], linewidth=1, label='AR')
    ax.plot(test_x, result[len(real_y) - 1:], 'g', linewidth=1, label='prediction')
    ax.set_xlabel('Days')
    ax.set_ylabel('Price, USD')
    ax.set_title('AR')
    plt.grid()
    plt.legend()
    plt.show()


def ma_method(real_x, real_y, test_x, test_y):
    ma_model = ARIMA(real_y, order=(0, 0, 5))
    ma_fit = ma_model.fit()
    result = ma_fit.predict(start=real_x[1], end=len(real_y) + len(test_y) - 1)

    rmse_test = np.sqrt(mean_squared_error(test_y, result[len(real_x) - 1:]))
    print(f'RMSE для модели MA = {rmse_test}')

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(real_x, real_y, linewidth=1, label='real')
    ax.plot(test_x, test_y, 'r', linewidth=1, label='real last 14 days')
    ax.plot(real_x, result[:len(real_y)], linewidth=1, label='MA')
    ax.plot(test_x, result[len(real_y) - 1:], 'g', linewidth=1, label='prediction')
    ax.set_xlabel('Days')
    ax.set_ylabel('Price, USD')
    ax.set_title('МА')
    plt.grid()
    plt.legend()
    plt.show()


def arma_method(real_x, real_y, test_x, test_y):
    arma_model = ARIMA(real_y, order=(7, 0, 7))
    arma_fit = arma_model.fit()
    result = arma_fit.predict(start=real_x[1], end=len(real_y) + len(test_y) - 1)

    rmse_test = np.sqrt(mean_squared_error(test_y, result[len(real_x) - 1:]))
    print(f'RMSE для модели ARMA = {rmse_test}')

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(real_x, real_y, linewidth=1, label='real')
    ax.plot(test_x, test_y, 'r', linewidth=1, label='real last 14 days')
    ax.plot(real_x, result[:len(real_y)], linewidth=1, label='ARMA')
    ax.plot(test_x, result[len(real_y) - 1:], 'g', linewidth=1, label='prediction')
    ax.set_xlabel('Days')
    ax.set_ylabel('Price, USD')
    ax.set_title('ARMA')
    plt.grid()
    plt.legend()
    plt.show()


# загрузка данных из файла
data = pd.read_excel(r'bitcoin.xlsx', sheet_name='bitcoin')
x_form = [0, ]
for i in range(1, len(data.iloc[:, 0])):
    x_form.append(x_form[-1] + abs((data.iloc[:, 0][i] - data.iloc[:, 0][i - 1]).days))
x = np.array(x_form)
y = data['Значение'].to_numpy()

real_x = x[:-13].copy()
real_y = y[:-13].copy()
test_x = x[-14:].copy()
test_y = y[-14:].copy()

moving_average_method(real_x, real_y, test_x, test_y, data)
exponential_smoothing_method(real_x, real_y, test_x, test_y)
trend_method(real_x, real_y, test_x, test_y)
ar_method(real_x, real_y, test_x, test_y)
ma_method(real_x, real_y, test_x, test_y)
arma_method(real_x, real_y, test_x, test_y)

