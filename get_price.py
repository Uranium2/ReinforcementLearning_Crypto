import csv

import pandas as pd
csv_input = pd.read_csv('coinbaseUSD_1D.csv')
csv_out = pd.read_csv('foo2.csv')
csv_out['Price'] = csv_input['Close']
csv_out.to_csv('output.csv', index=False)