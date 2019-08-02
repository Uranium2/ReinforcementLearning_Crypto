import pandas as pd

df=pd.read_csv('coinbaseUSD_1min_clean.csv')


FINDING MAX AND MIN
maxi = df['Close'].max()
mini = df['Close'].min()
maxi = 19891.99
mini = 0.06
print(maxi)
print(mini)
