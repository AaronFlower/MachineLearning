## 处理异常数据
# handling of erroneous/outlier data.

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

# rolling examples

bars = {'value': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}
df = pd.DataFrame(bars)
df['MEAN'] = df['value'].rolling(window=2).mean()
df['STD'] = df['value'].rolling(window=2).std()
print(df)

df['MEAN'] = df['value'].rolling(window=1).mean()
df['STD'] = df['value'].rolling(window=1).std()
print(df)

df['MEAN'] = df['value'].rolling(window=3).mean()
df['STD'] = df['value'].rolling(window=3).std()
print(df)

df['MEAN'] = df['value'].rolling(window=10).mean()
df['STD'] = df['value'].rolling(window=10).std()
print(df)

## bridge data

print('Bridge Data Example:')

bridge_height = {'meters': [10.26, 10.31, 10.27, 10.22, 10.23, 6212.42, 10.28, 10.25, 10.31]}
df = pd.DataFrame(bridge_height)
df['MEAN'] = df['meters'].rolling(window=2).mean()
df['STD'] = df['meters'].rolling(window=2).std()
df_std = df.describe()
print(df)
print(df.describe())
df_std = df.describe()['meters']['std']
# delete outlier
df = df[ (df['STD'] < df_std) ]
print(df)

df['meters'].plot()
plt.show()