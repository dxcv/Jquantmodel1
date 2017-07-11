# 单因子回测策略
import pandas as pd
import numpy as np


# 从文件读取数据
def get_data(data):
    df = pd.read_csv('E:\quant\git\PythonApplication1\PythonApplication1\panel\%s.csv' % data)
    indexed_df = df.set_index('Unnamed: 0')
    indexed_df.index.name = data
    return indexed_df


# 根据所选因子，获得ＴＯＰ的股票名称
def top_name(df, percentage):
    stock = {}
    for date in df.index:
        row = df.loc[date]
        row = row.sort_values()
        breakpoint = int(np.floor(row.count() * percentage))
        select_value = row[:breakpoint]
        stock[date] = list(select_value.index)
    return stock


########################################################################################################################
########################################################################################################################
# 主程序

percentage = 0.2
open_price = get_data('open')
close_price = get_data('close')
day_return = close_price - open_price
top_open = top_name(open_price, percentage)
# 输出股票去dataframe
holding_stock = pd.DataFrame.from_dict(top_open, orient='index')
# holding_stock.to_csv('holding_stock.csv')
for stock in holding_stock.loc['2015-02-05']:
    print(stock)

# 选出对应股票的 forward return
dates = list(top_open.keys())

print(top_open)
