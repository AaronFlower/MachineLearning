## 使用 Mapping 函数及打标签 labels.

import pandas as pd
import quandl
import numpy as np

def get_clean_data():
	''' 数据 Normaliztion 及清洗 '''
	housing_data = pd.read_pickle('HPI.pickle')
	housing_data.rename(columns={'Value': 'USA_HPI'}, inplace=True)
	## HPI 的数据计算是从初始化的增加率计算的。
	print(housing_data.dropna().head())

	## 计算环比数据, 即 normalize 数据。
	housing_data = housing_data.pct_change()

	## 新增一列下个月的数据，有助于对比和生成 label
	housing_data['USA_HPI_next'] = housing_data['USA_HPI'].shift(-1)

	## 处理 -inf, inf 数据，先将 inf 都替换成 nan, 然后再统一删除 nan.
	housing_data.replace([np.inf, -np.inf], np.nan, inplace=True)
	housing_data.dropna(inplace=True)
	return housing_data

def create_label(cur_hpi, next_hpi):
	''' HPI 增长则标记为 1 , 反之标记为 0'''
	if next_hpi > cur_hpi:
		return 1
	else:
		return 0

def generate_labels(df):
	''' 
		生成数据集的 labels 
		利用 map 函数， map 函数原型： map(function_to_map, param1, param2)
	'''
	df['label'] = list(map(create_label, df['USA_HPI'], df['USA_HPI_next']))

housing_data = get_clean_data()
generate_labels(housing_data)
print(housing_data)


