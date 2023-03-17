import math
import matplotlib.pyplot as plt


def fun(S, E, I, t, lamb):
    p = t * lamb / 60

    P_zero = 0
    for n in range(S):
        P_zero += math.pow(p, n) / math.factorial(n)
    P_zero = pow(P_zero, -1)

    q = 1 - math.pow(p, S) / math.factorial(S) * P_zero
    A = lamb * q
    res = I * A - S * E
    return res


def graph(E, I, t, lamb):
    plt.figure(figsize=(7, 10))
    plt.xlabel('Количество полос')
    plt.ylabel('Прибыль')
    plt.grid()
    max_res = -100
    max_S = 0

    for S in range(1, 11):
        cur = fun(S, E, I, t, lamb)
        if max_res < cur:
            max_res = round(cur, 2)
            max_S = S
        plt.plot(S, cur, 'o', c='red')

    plt.title(f'Доход от обслуживания заявки = {I}\nРасход на содержание канала = {E}\n'
              f'Время обслуживания = {t}, Интенсивность = {lamb}\n\n'
              f'Максимальная прибыль = {max_res} при количестве полос = {max_S}')
    plt.show()


if __name__ == "__main__":
    graph(E=2, I=5, t=10, lamb=15)
    graph(E=1, I=6, t=15, lamb=10)
    graph(E=3, I=8, t=20, lamb=15)
    graph(E=1, I=3, t=5, lamb=10)
