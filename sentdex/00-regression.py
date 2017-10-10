#sklearn, quandl, pandas

import pandas as pd
import quandl
import math

api_key = open('../quandl-token.txt', 'r').read()
quandl.ApiConfig.api_key = api_key


def grap_fresh_data():
	''' 将拉取的数据存储下来 '''
	quandl.ApiConfig.api_key = api_key
	df = quandl.get('WIKI/GOOGL')
	df.to_pickle('GOOGLE-2017-10-09.pickle')

def load_data():
	'''读取原始数据'''
	df = pd.read_pickle('GOOGLE-2017-10-09.pickle')
	return df

def get_adjusted_data(df):
	'''我们 Adjusted Price 对我们更重要'''
	df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
	# 把一天内的 volatility 考虑进来，
	# high - low percent change
	df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
	df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
	df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
	return df
# grap_fresh_data() # 2017-10-09

df = load_data()
df = get_adjusted_data(df)
print(df.head())


# forecast_col = 'Adj. Close'
# df.fillna(-99999, inplace=True)

# # test samples. 10% 的测试用例
# forcast_out = int(math.ceil(0.01 * len(df)))

# df['label'] = df[forecast_col].shift(-forcast_out)
# df.dropna(inplace=True)

# print(df)
# 

