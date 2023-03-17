import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyEMD import EMD

# загрузка данных из Excel файла
data = pd.read_excel('нефть-brent.xlsx', sheet_name='нефть brent', header=None)
data = data[::-1]
data = np.array(data[1])

plt.plot(data)
plt.title('Исходный график')
plt.show()

# проведение EMD разложения
emd = EMD()
imfs = emd.emd(data)

# построение графиков для каждой интрузивной моды
for i, imf in enumerate(imfs):
    # plt.subplot(len(imfs), 1, i+1)
    plt.plot(imf)
    plt.title('IMF ' + str(i+1))
    plt.show()

itog = sum(imfs)
plt.plot(itog)
plt.title('Восстановленный график')
plt.show()

