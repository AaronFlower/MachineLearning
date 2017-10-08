## 使用 Pick 序列化对象
# 注意： pickle 是不安全的，Never unpickle data received from an untrusted or unauthenticated source.
# 1. 使用 Python 内置 pickle 进行序列化。
# 2. 使用 pandas 的 pickle 进行序列化。

import quandl
import pandas as pd
import pickle

# 初始化 50 州
def list_states():
	states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')
	return states[0][0][1:]
# 获取初始化数据
def grab_initial_data():
	main_df = pd.DataFrame()
	api_key = open('quandl-token.txt', 'r').read()
	states = list_states()
	for state in states:
		query = "FMAC/HPI_" + state
		df = quandl.get(query, authtoken=api_key, start_date='2001-01-31')
		df.columns = [state]
		if main_df.empty:
			main_df = df
		else:
			main_df = main_df.join(df)

	return main_df

def dumpToPickle(df):
	pickle_out = open('py-states-data.pickle', 'wb')
	pickle.dump(df, pickle_out)
	pickle_out.close()

def loadPickle():
	pickle_in = open('py-states-data.pickle', 'rb')
	df = pickle.load(pickle_in)
	pickle_in.close()
	return df

# states_df = grab_initial_data()
# dumpToPickle(states_df)
states_df = loadPickle()
print(states_df.head())

# 使用 pandas 自带的 pickle
states_df.to_pickle('pd-inner.pickle')
states_df2 = pd.read_pickle('pd-inner.pickle')
print(states_df2.head())

