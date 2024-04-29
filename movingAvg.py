from cProfile import label
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


csv = pd.read_csv('FMS K9 TEST#41_Cropped.csv', names=['oil'])
csv['seconds'] = csv.index
csv['seconds'] = csv['seconds']/60
csv['oil'] = csv['oil']/1000

window = 30

y = csv['oil'].rolling(window=window).mean()
plt.plot(csv['seconds'],csv['oil'],label='Original')
plt.plot(csv['seconds'],y,label=f'MvAVG {window}pts')
plt.xlabel('segundos')
plt.ylabel('óleos')
plt.legend()
plt.show() 