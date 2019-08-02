import pandas as pd
import numpy
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df = pd.read_csv('coinbaseUSD_1min_clean.csv')
df_scaled = scaler.fit_transform(df)
numpy.savetxt("foo.csv", df_scaled, delimiter=",")
