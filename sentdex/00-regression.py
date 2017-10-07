#sklearn, quandl, pandas

import pandas as pd
import quandl
import math

quandl.ApiConfig.api_key = '8bnp57r24GpxbF-aCf1A'


#dataframe
df = quandl.get("WIKI/GOOGL")
# print(df.head())
# 
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
print(df.head())

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

# test samples. 10% 的测试用例
forcast_out = int(math.ceil(0.01 * len(df)))

df['label'] = df[forecast_col].shift(-forcast_out)
df.dropna(inplace=True)

print(df.head())
