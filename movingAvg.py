import pandas as pd
import matplotlib.pyplot as plt

def movingAVG(filename = 'No oil_Cropped.csv'):
    csv = pd.read_csv(filename, names=['oil'])
    csv['seconds'] = csv.index
    csv['seconds'] = csv['seconds']/60
    csv['oil'] = csv['oil']/1000
    
    _,[ax0,ax1] = plt.subplots(2,1,figsize = (10,8),sharex=True)
    
    
    window = 30
    y = csv['oil'].rolling(window=window).mean()
    ax0.plot(csv['seconds'],csv['oil'],label='Original')
    ax0.plot(csv['seconds'],y,label=f'MvAVG {window}pts')
    ax0.legend()
    ax0.set_ylabel('óleos')

    window = 60
    y = csv['oil'].rolling(window=window).mean()
    ax1.plot(csv['seconds'],csv['oil'],label='Original')
    ax1.plot(csv['seconds'],y,label=f'MvAVG {window}pts')
    ax1.legend()
    ax1.set_ylabel('óleos')
    ax1.set_xlabel('Segundos')

    plt.show()
# movingAVG()
