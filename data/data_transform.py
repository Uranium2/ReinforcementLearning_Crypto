import numpy as np 
import pandas as pd
import datetime
import os

data = pd.read_csv("coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv")


def label_var(row):
    return (row['Close'] - row['Open']) / row['Open'] * 100

def sign(row):
    if row['Variation'] > 0:
        return 1
    return -1

data.Timestamp = pd.to_datetime(data.Timestamp, unit='s')
data['Volume_(BTC)'].fillna(value=0, inplace=True)
data['Volume_(Currency)'].fillna(value=0, inplace=True)
data['Weighted_Price'].fillna(value=0, inplace=True)

data['Open'].fillna(method='ffill', inplace=True)
data['High'].fillna(method='ffill', inplace=True)
data['Low'].fillna(method='ffill', inplace=True)
data['Close'].fillna(method='ffill', inplace=True)
data.index = data.Timestamp

data = data.resample('Y').agg({'Open': 'first', 
                'High': np.max,
                'Low': np.min,
                'Close': 'last',
                'Volume_(BTC)': np.sum,
                'Volume_(Currency)': np.sum,
                'Weighted_Price': np.mean})


data['Variation'] = data.apply(lambda row: label_var(row), axis=1)
data['Sign'] = data.apply(lambda row: sign(row), axis=1)

data.to_csv("coinbaseUSD_1Y.csv")

print(data)