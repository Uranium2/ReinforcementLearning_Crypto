import pandas as pd
import numpy
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()
scaler2 = StandardScaler()


df = pd.read_csv('coinbaseUSD_1D.csv')



df_scaled = scaler.fit_transform(df)
print(df_scaled)
df_normed = scaler2.fit_transform(df_scaled)
print(df_normed)

numpy.savetxt("foo2.csv", df_normed, delimiter=",")
