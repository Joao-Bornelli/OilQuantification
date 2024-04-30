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

fig,[ax0,ax1] = plt.subplots(2,1,figsize = (10,8),sharex=True)


ax0.plot(csv['seconds'],csv['oil'],label='Original')
ax0.plot(csv['seconds'],y,label=f'MvAVG {window}pts')
# ax0.set_xlabel('segundos')
ax0.legend()
ax0.set_ylabel('óleos')

y = csv['oil'].rolling(window=window*2).mean()
ax1.plot(csv['seconds'],csv['oil'],label='Original')
ax1.plot(csv['seconds'],y,label=f'MvAVG {window*2}pts')
# ax0.set_xlabel('segundos')
ax1.legend()
ax1.set_ylabel('óleos')

ax1.set_xlabel('Segundos')

plt.show()