# -*- encoding: utf-8 -*-
## 使用 Mapping 函数及打标签 labels.
#

import pandas as pd
import quandl
import numpy as np
# sklearn, scikit-learn 
# svm 训练器 
from sklearn import svm # 数据预处理
from sklearn import preprocessing 
# 交叉验证
from sklearn import cross_validation 

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

# features X
X = np.array(housing_data.drop(['label', 'USA_HPI_next'], axis=1)) # drop axis = 1 表示删除的是列， 默认是 indexes 行。
print('Initial Features:')
print(X)
# preprocessing 预处理一下，feature 的值保持在 [-1, 1] 之间，计算会比较精确。
X = preprocessing.scale(X)
print('After preprocessing:')
print(X)

# labels 
y = np.array(housing_data['label'])

# 区分训练样本及测试样本
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# 给出分类器, svm, SVC: Support Vector Classification.
classifier = svm.SVC(kernel='linear')
# 拟合数据 
classifier.fit(X_train, y_train)

# 输出测试精确度
print('SVM accuracy:', classifier.score(X_test, y_test))

# 预测数据，预测数据也是从之前数据中取的。 
predict_data = np.array(housing_data.tail(5).drop(['label', 'USA_HPI_next'], axis=1))
predict_data = preprocessing.scale(predict_data)
print('Predict Result:', classifier.predict(predict_data))
print('Tail 5 labels:', housing_data.tail(5)['label'])

# 看出学习效果并不是很好，为什么那？ features 关联度太小了吧。

