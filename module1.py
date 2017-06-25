#单因子回测策略
import pandas as pd
import numpy as np
df=pd.read_csv('E:\quant\git\PythonApplication1\PythonApplication1\panel\open.csv')
percentage=0.2
indexed_df=df.set_index('Unnamed: 0')
indexed_df.index.name='Date'

#获得ＴＯＰ的股票名称
stock={}
for date in indexed_df.index:
    row=indexed_df.loc[date]
    row=row.sort_values()
    breakpoint=int(np.floor(row.count()*percentage))
    list=row[:breakpoint]
    stock[date]=list.index
print(stock)

