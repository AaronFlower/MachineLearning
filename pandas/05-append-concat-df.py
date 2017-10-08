## DataFrame concat and append
import pandas as pd

df1 = pd.DataFrame({
	'HPI':[80,85,88,85],
	'Int_rate':[2, 3, 2, 2],
	'US_GDP_Thousands':[50, 55, 65, 55]
	}, index = [2001, 2002, 2003, 2004]
)

df2 = pd.DataFrame({
	'HPI':[80,85,88,85],
	'Int_rate':[2, 3, 2, 2],
	'US_GDP_Thousands':[50, 55, 65, 55]
	}, index = [2005, 2006, 2007, 2008]
)

df3 = pd.DataFrame({
	'HPI':[80,85,88,85],
	'Int_rate':[2, 3, 2, 2],
	'Low_tier_HPI':[50, 52, 50, 53]
	}, index = [2001, 2002, 2003, 2004])

print(df1)
print(df2)
## concat
print('pd.concat([df1, df2]):')
concat = pd.concat([df1, df2])
# 有不同的列并且 index 相同, 会出现 Not a nubmer
print('pd.concat([df1, df2, df3]):')
print(pd.concat([df1, df2, df3]))

## append
print('after append df2')
df4 = df1.append(df2)
print(df4)

# 同样会出现 Not a number.
print('after append df3')
df4 = df1.append(df3)
print(df4)

# append one row
s = pd.Series([88, 2, 55])
print(s)
s = pd.Series([88, 2, 55], index=['HPI', 'Int_rate', 'US_GDP_Thousands'])
print(s)
df4 = df1.append(s, ignore_index=True)
print(df4)
