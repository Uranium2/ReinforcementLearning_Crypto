import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from multiprocessing import Pool


data = pd.read_csv("coinbaseUSD_1D.csv")
 
x = data['Timestamp']
y = data['Close']




def diplay_graphs(x, y):
    h = ['Close', 'Action']
    for i in range(5):
        data2 = pd.read_csv("log_actions_" + str(i) + ".csv", names=h)
        actions = data2['Action']

        xBuy = []
        yBuy = []

        for i in range(len(actions)):
            if actions[i] == 'BUY':
                xBuy.append(x[i])
                yBuy.append(10000)


        xSell = []
        ySell = []

        for i in range(len(actions)):
            if actions[i] == 'SELL':
                xSell.append(x[i])
                ySell.append(10000)
        plt.figure(i)
        plt.plot(x,y)
        green = (0, 1, 0, 0.1)
        red = (1, 0, 0, 0.1)
        plt.bar(xBuy, yBuy, color=green)
        plt.bar(xSell, ySell, color=red)

        plt.xticks([])
    plt.show()


diplay_graphs(x, y)