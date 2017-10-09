## 知道了房价走势，再考虑下利率，Interest Rates.
# Mortgate reate, 按揭利率。

import pandas as pd
import quandl
import pickle

api_key = open('quandl-token.txt', 'r').read()

def loadData():
	'''获取初始化数据'''
	pickle_in = open('py-states-data.pickle', 'rb')
	df = pickle.load(pickle_in) 
	return df

# 初始化 50 州
def list_states():
	states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')
	return states[0][0][1:]

def HPI_dataProcess():
	df = loadData()
	states = list_states()
	for state in states:
		df[state] = (df[state] - df[state][0]) / df[state][0] * 100.0

	return df

def HPI_Benchmark():
  df = quandl.get("FMAC/HPI_USA", authtoken=api_key)
  df["United States"] = (df["Value"]-df["Value"][0]) / df["Value"][0] * 100.0
  df.rename(columns={'United States':'US_HPI'}, inplace=True)
  return df

def grap_30yr_mortgage():
	''' 30 年按揭利率 '''
	df = quandl.get('FMAC/MORTG', trim_start='1975-01-01', authtoken=api_key)
	df['Value'] = (df['Value'] - df['Value'][0]) / df['Value'][0] * 100.0
	df = df.resample('M').mean() # 按月重新采样，日期与 HPI 的日期保持一致。
	return df

def sp500_data():
	''' 标准普尔 500， The Standard & Poor's 500,  S&P 500, or just  S&P '''
	# YAHOO 的数据已经挂了。
	df = quandl.get("YAHOO/INDEX_GSPC", trim_start="1975-01-01", authtoken=api_key)
	df["Adjusted Close"] = (df["Adjusted Close"]-df["Adjusted Close"][0]) / df["Adjusted Close"][0] * 100.0
	df=df.resample('M').mean()
	df.rename(columns={'Adjusted Close':'sp500'}, inplace=True)
	return df

def gdp_data():
	''' GDP: gross domestic product '''
	df = quandl.get("BCB/4385", trim_start="1975-01-01", authtoken=api_key)
	print('gdp_data')
	df["Value"] = (df["Value"]-df["Value"][0]) / df["Value"][0] * 100.0
	df=df.resample('M').mean()
	df.rename(columns={'Value':'GDP'}, inplace=True)
	return df

def us_unemployment():
	''' USA unemployment '''
	df = quandl.get("ECPI/JOB_G", trim_start="1975-01-01", authtoken=api_key)
	df["Unemployment Rate"] = (df["Unemployment Rate"]-df["Unemployment Rate"][0]) / df["Unemployment Rate"][0] * 100.0
	df=df.resample('1D').mean()
	df=df.resample('M').mean()
	return df

# tutorial 14
HPI_bench = HPI_Benchmark()
m30 = grap_30yr_mortgage()
m30.columns = ['M30']
# sp500 = sp500_data()
## GDP 的数据是从 1990 年开始的。
gdp = gdp_data()
# unemployment = us_unemployment()
HPI = HPI_bench.join([m30, gdp])
HPI.to_pickle('HPI.pickle')
print(HPI.dropna())
print(HPI.corr().head())
