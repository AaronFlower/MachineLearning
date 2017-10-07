import pandas as pd 
import datetime
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from matplotlib import style

start = datetime.datetime(2016, 1, 1)
end = datetime.datetime(2017, 10, 1)

# to pull data from the internet
df = web.DataReader('XOM', 'yahoo', start, end)
print(df.head())

style.use('fivethirtyeight')
df['High'].plot()
plt.legend()
plt.show()


