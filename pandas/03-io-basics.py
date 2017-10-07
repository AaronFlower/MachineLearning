# pandas io 主要用来处理文件的输入输出
# pandas 支持文件类型有, csv, json, html, excel, sql 等
# http://pandas.pydata.org/pandas-docs/version/0.20/io.html#io-read-csv-table

import pandas as pd
df = pd.read_csv('NBSC-A0L090D_A.csv')
print(df.head())
# 将读取的数据 output to csv
# 首先要去掉默认的 index
df.set_index('Date', inplace = True)
df.to_csv('Shanghai_PERatio.csv') # 上海市盈率 Price to Earnings Ratio
df = pd.read_csv('Shanghai_PERatio.csv')
print('Shanghai_PERatio CSV')
print(df.head())

# 从 csv 文件读取可指定 index column 以及重命名 column
df = pd.read_csv('Shanghai_PERatio.csv', index_col = 0)
print(df.head())
df.columns = ['Shanghai_PERatio']
print(df.head())

# 保存时可以去掉 header
df.to_csv('PE_no_headers.csv', header = False)

# 保存时去掉了 header， 那么读取时就可以加上 header
df = pd.read_csv('PE_no_headers.csv', names = ['Date', 'Shanghai_PERatio'], index_col = 0)
print(df.head())
df.rename(columns = {'Shanghai_PERatio': 'Shanghai_PER'}, inplace = True)
print(df.head())
# save html 文件
df.to_html('SH_PER.html')

