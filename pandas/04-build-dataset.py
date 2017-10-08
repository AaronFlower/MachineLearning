# pip install quandl
# house price index
import quandl
import pandas as pd

api_key = open('quandl-token.txt', 'r').read()
df = quandl.get("FMAC/HPI_CA", authtoken=api_key, start_date="2001-01-31")
print(df.head())
# Mac 访问 https 需要先安装: /Applications/Python 3.6/Install Certificates.command
states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')
# print(states[0][0])

# states[0][0][1:] 去掉 abbreviation header
for abbr in states[0][0][1:]:
	print("FMAC/HPI_" + abbr)


