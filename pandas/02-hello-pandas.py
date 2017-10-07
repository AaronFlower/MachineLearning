import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')

web_stats = {
	'Day': [1, 2, 3, 4, 5, 6],
	'Visitors': [43, 53, 45, 54, 54, 23],
	'Bounce_Rate': [65, 70, 34, 56, 54, 62]
}

df = pd.DataFrame(web_stats)
# print(df)
# print(df.head())
# print(df.head(2))
# print(df.tail())
# print(df.tail(2))
# print(df)
# return a new DataFrame
# print(df.set_index('Day'))
# print(df)

# df2 = df.set_index('Day')
# print(df2.head())
# 直接替换而不是返回一个新的。
# df.set_index('Day', inplace=True)
# print(df.head())
# 
# refrence one column
print(df['Visitors'])
print(df.Visitors)

# reference multiple columns
print(df[['Visitors', 'Bounce_Rate']])

# convert to list
print(df.Visitors.tolist())

# error
# print(df[['Visitors', 'Bounce_Rate']].tolist())
print(np.array(df[['Visitors', 'Bounce_Rate']]))

# plot a single row
df['Visitors'].plot()
plt.show()
df2 = df.set_index('Day')

df2.plot()
plt.show()
