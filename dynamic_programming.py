def dynamic(n):
    F = [0 for i in range(n)]
    F[0] = 2
    F[1] = 4
    F[2] = 8
    F[3] = 15
    for i in range(4, n):
        F[i] = F[i-1] + F[i-2] + F[i-3] + F[i-4]

    return F[n-1]

n = int(input('Введите число проверок: '))
print(f'Число возможных проверок: {dynamic(n)}')