import pandas as pd

df=pd.read_csv('coinbaseUSD_1D.csv')


#FINDING MAX AND MIN
maxi = df['Close'].max()
mini = df['Close'].min()
print(maxi)
print(mini)
