## 使用 Pick 序列化对象
# 注意： pickle 是不安全的，Never unpickle data received from an untrusted or unauthenticated source.
# 1. 使用 Python 内置 pickle 进行序列化。
# 2. 使用 pandas 的 pickle 进行序列化。

import quandl
import pandas as pd
import pickle
import matplotlib.pyplot as plt

api_key = open('quandl-token.txt', 'r').read()


# 初始化 50 州
def list_states():
	states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')
	return states[0][0][1:]
# 获取初始化数据
def grab_initial_data():
	main_df = pd.DataFrame()
	states = list_states()
	for state in states:
		query = "FMAC/HPI_" + state
		df = quandl.get(query, authtoken=api_key)
		df.columns = [state]
		if main_df.empty:
			main_df = df
		else:
			main_df = main_df.join(df)

	return main_df

def grab_HPI_benchmark():
	''' 获取全国数据 '''
	df = quandl.get('FMAC/HPI_USA', authtoken=api_key)
	df.columns = ['United States']
	df['United States'] = (df['United States'] - df['United States'][0]) / df['United States'][0] * 100
	return df

def dumpToPickle(df):
	pickle_out = open('py-states-data.pickle', 'wb')
	pickle.dump(df, pickle_out)
	pickle_out.close()

def loadPickle():
	pickle_in = open('py-states-data.pickle', 'rb')
	df = pickle.load(pickle_in)
	pickle_in.close()
	return df

def initDataProcess():
	''' HPI 数据根据第一年的数据作下预处理 '''
	HPI_data = loadPickle()
	states = list_states()
	for state in states:
		HPI_data[state] = (HPI_data[state] - HPI_data[state][0]) / HPI_data[state][0] * 100

	return HPI_data

def plotUSA_HPI():
	''' 绘制整个美国的 HPI '''
	fig = plt.figure()
	ax1 = plt.subplot2grid((1, 1), (0, 0))

	HPI_data = initDataProcess()
	benchmark = grab_HPI_benchmark()
	HPI_data.plot(ax=ax1)
	benchmark.plot(color='k', ax=ax1, linewidth=5)

	plt.legend().remove()
	plt.show()

def plotStateYearSample(state):
	''' 绘制州的年数据 '''
	# http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
	HPI_data = initDataProcess()
	fig = plt.figure()
	ax1 = plt.subplot2grid((1, 1), (0, 0))
	State1yr = HPI_data[state].resample('A').mean() # A : Annual, A	year end frequency
	print(State1yr.head())
	HPI_data[state].plot(ax=ax1, label=state + ' HPI')
	State1yr.plot(ax=ax1, color='k', label = state + ' yearly HPI')
	plt.legend(loc=4)
	plt.show()

plotStateYearSample('CA')
plotStateYearSample('TX')