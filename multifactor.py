# 多因子选股模型
# 先导入所需要的程序包
# import tushare as ts
import statsmodels.api as sm
from statsmodels import regression
import datetime
import numpy as np
import pandas as pd
import time
from jqdata import *
from pandas import Series, DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

'''
================================================================================
总体回测前
================================================================================
'''


# 本程序实现了东方证券机器学习因子库中的  {“乒乓球反转因子” PPReversal, CGO Capital Gains Overhang
#   TO 流通股本日均换手率, EPS_GR(EPS Growth Ratio)每股收益增长率（年度同比）
#   Revenue Growth Ratio营业收入增长率 季度同比 , 净利润增长率季度同比 net_profit Growth Ratio
#   非流动性因子illiquidity factor(ILLIQ) , Fama-French 三因子、五因子模型、
#   市值因子}


# 总体回测前要做的事情
def initialize(context):
    set_params()  # 1 设置策参数
    set_variables()  # 2 设置中间变量
    set_backtest()  # 3 设置回测条件


# 1 设置策略参数
def set_params():
    # 单因子测试时g.factor不应为空
    g.factor = 'AllFactor'  # 当前回测的单因子
    # g.shift = 21                    # 设置一个观测天数（天数）
    g.stock_num = 50.0  # 设置买卖股票数目，坑爹的python2的除法，一定要设置成浮点数。
    # 设定选取sort_rank： True 为最大，False 为最小
    g.sort_rank = True
    # 多因子合并称DataFrame，单因子测试时可以把无用部分删除提升回测速度
    # 定义因子以及排序方式，默认False方式为降序排列，原值越大sort_rank排序越小
    g.factors = {'AllFactor': False}
    # 计算因子均值和相关系数的调仓周期数,不需要的g.periods的时候不用管，程序会自动忽略
    g.periods = 2
    # 各周期的训练数据周期数
    g.period_all = [1, 2, 2]
    # 每日调仓比例，控制换手率
    g.stock_per_day = 0.1
    # 选择因子加权方法
    # linear_regression_choose=False       # 线性回归 0
    # optimal_choose=True           # 最优化  1
    # randomtree_choose   #随机森林 2
    g.contrl_choose_name = ['linear_regression_choose', 'optimal_choose', 'randomtree_choose']
    # 控制选股方法
    g.contrl_choose = [2, 2, 2]
    # 是否线性回归得到因子加权系数
    g.regression_method = False
    # 是否计算因子值
    g.cal_factor = True
    # 是否行业中性化处理
    g.Industry_Neutral = False
    # 是否去极值
    g.Drop_Extremum = True
    # 是否标准化 最优化需要标准化
    g.normalization = False
    # 是否去掉创业板True则去掉
    g.No_GEM = True
    # 是否去掉上市不满三个月的新股True则去掉
    g.No_new_stock = True
    # 按月回测因子名称,最后一个因子为涨跌幅 不能去掉 也不能改变位置
    list_factor_month = ['MTM3', 'BP', 'MTM1', 'EPS_GR', 'RGR', 'EP', 'TRC', 'NPGR', 'PS', 'VARC', 'FF', 'CMC',
                         'Price_change_M']
    list_factor_week = ['PPReversal', 'ILLIQ', 'Price_change_W']
    list_factor_day = ['TO', 'Price_change_D']  # ,'alpha_55'
    # 写到一起
    g.list_factor_all = [list_factor_month, list_factor_week, list_factor_day]
    g.p_change = ['Price_change_M', 'Price_change_W', 'Price_change_D']
    # 参与回测的因子名称，不添加因子名称则不参与回测，MTM1同时也是一个月涨跌幅，不能去掉
    # 设置全局变量用于保存计算结果
    # 定义全局变量 保存多因子信息
    # 去极值、（标准化）后的因子值
    g.panel_factor_month = pd.Panel()  # 按月
    g.panel_factor_week = pd.Panel()  # 按周
    g.panel_factor_day = pd.Panel()  # 按天
    # 三个写到一起
    g.list_panel_set = [g.panel_factor_month, g.panel_factor_week, g.panel_factor_day]
    # 定义全局变量存储IC值
    g.df_IC_month = DataFrame(index=list_factor_month)
    g.df_IC_week = DataFrame(index=list_factor_week)
    g.df_IC_day = DataFrame(index=list_factor_day)
    # 三个写到一起
    g.list_df_IC_set = [g.df_IC_month[:-1], g.df_IC_week[:-1], g.df_IC_day[:-1]]
    # 计数全局变量
    g.count_M = 0  # 月调仓
    g.count_w = 0  # 周调仓
    g.count_D = 0  # 日调仓
    # 三个写到一起
    g.list_count_set = [g.count_M, g.count_w, g.count_D]
    # 总共的交易天数
    g.days_num = 0
    # 按月调仓选出的股票数目
    g.stock_num_M = 500
    # 按周调仓选出的股票数目
    g.stock_num_W = 150
    # 最终每日交易的股票数目
    g.num_stocks_D = 50
    # 保存机器学习模型
    g.ml_m = []
    g.ml_w = []
    g.ml_d = []
    g.machinelearning = [g.ml_m, g.ml_w, g.ml_d]
    # 保存机器学习的训练数据
    g.x_m = []
    g.x_w = []
    g.x_d = []
    g.y_m = []
    g.y_w = []
    g.y_d = []
    g.x_all = [g.x_m, g.x_w, g.x_d]
    g.y_all = [g.y_m, g.y_w, g.y_d]
    g.holding_list_M = []
    g.holding_list_W = []


# 2 设置中间变量
def set_variables():
    g.feasible_stocks = []  # 当前可交易股票池
    g.if_trade = False  # 当天是否交易


# 3 设置回测条件
def set_backtest():
    set_benchmark('399300.XSHE')  # 设置为基准 沪深300
    set_option('use_real_price', True)  # 用真实价格交易
    log.set_level('order', 'error')  # 设置报错等级


'''
================================================================================
每天开盘前
================================================================================
'''


# 每天开盘前要做的事情
def before_trading_start(context):
    # 计算交易天数，这个函数每天运行
    g.days_num += 1
    # 5 设置每天的可行股票池：获得当前开盘的股票池并剔除当前或者计算样本期间停牌的股票
    list_stocks = list(get_all_securities(['stock']).index)
    # 去掉创业板
    if g.No_GEM:
        list_stocks = filter_gem_stock(list_stocks)
    # 去掉上市不满三个月的新股
    if g.No_new_stock:
        list_stocks = filter_new_stock(context, list_stocks)
    g.feasible_stocks = set_feasible_stocks(list_stocks, context)
    # 6 设置手续费与手续费
    set_slip_fee(context)


# 剔除上市不满3个月的股票
def filter_new_stock(context, stock_list):
    tmpList = []
    for stock in stock_list:
        days_public = (context.current_dt.date() - get_security_info(stock).start_date).days
        # 上市未超过三个月
        if days_public > 93:
            tmpList.append(stock)
    return tmpList


# 过滤掉创业板
def filter_gem_stock(stock_list):
    return [stock for stock in stock_list if stock[0:3] != '300']


# 5
# 设置可行股票池
# 过滤掉当日停牌的股票,且筛选出前days天未停牌股票
# 输入：stock_list为list类型,样本天数days为int类型，context（见API）
# 输出：list=g.feasible_stocks
def set_feasible_stocks(stock_list, context):
    # 得到是否停牌信息的dataframe，停牌的1，未停牌得0
    suspened_info_df = get_price(list(stock_list),
                                 start_date=context.current_dt,
                                 end_date=context.current_dt,
                                 frequency='daily',
                                 fields='paused'
                                 )['paused'].T
    # 过滤停牌股票 返回dataframe
    unsuspened_index = suspened_info_df.iloc[:, 0] < 1
    # 得到当日未停牌股票的代码list:
    unsuspened_stocks = suspened_info_df[unsuspened_index].index
    current_data = get_current_data()
    return [stock for stock in unsuspened_stocks if not
    current_data[stock].is_st and
            'ST' not in current_data[stock].name and
            '*' not in current_data[stock].name and
            '退' not in current_data[stock].name]


# 6 根据不同的时间段设置滑点与手续费
def set_slip_fee(context):
    # 将滑点设置为0
    set_slippage(FixedSlippage(0))
    # 根据不同的时间段设置手续费
    dt = context.current_dt

    if dt > datetime.datetime(2013, 1, 1):
        set_commission(PerTrade(buy_cost=0.0003,
                                sell_cost=0.0013,
                                min_cost=5))

    elif dt > datetime.datetime(2011, 1, 1):
        set_commission(PerTrade(buy_cost=0.001,
                                sell_cost=0.002,
                                min_cost=5))

    elif dt > datetime.datetime(2009, 1, 1):
        set_commission(PerTrade(buy_cost=0.002,
                                sell_cost=0.003,
                                min_cost=5))

    else:
        set_commission(PerTrade(buy_cost=0.003,
                                sell_cost=0.004,
                                min_cost=5))


'''
================================================================================
每天交易时
================================================================================
'''


def handle_data(context, data):
    # 按月频因子筛选股票
    if (g.days_num - 1) % 20 == 0:
        # 8 获得买入卖出信号，输入context，输出股票列表list
        # 字典中对应默认值为false holding_list筛选为true，则选出因子得分最大的
        # num记录月周日频调仓
        g.holding_list_M = get_stocks(g.feasible_stocks,
                                      context,
                                      g.factors,
                                      asc=g.sort_rank,
                                      factor_name=g.factor,
                                      num=0)
        if len(g.holding_list_M) != 0:
            g.holding_list_M = g.holding_list_M[:g.stock_num_M]
    # 按周频因子筛选股票
    if (g.days_num - 1) % 5 == 0 or (g.days_num - 1) % 20 == 0:
        # 去掉停牌的股票
        g.holding_list_M = [stock for stock in g.holding_list_M if stock in g.feasible_stocks]
        stocks_wait_spilt_w = get_stocks(g.feasible_stocks,
                                         context,
                                         g.factors,
                                         asc=g.sort_rank,
                                         factor_name=g.factor,
                                         num=1)
        if len(stocks_wait_spilt_w) != 0:
            g.holding_list_W = [stock for stock in stocks_wait_spilt_w if stock in g.holding_list_M][:g.stock_num_W]
            # g.holding_list_W=g.holding_list_W[:g.stock_num_W]

    # 按日频因子筛选股票
    # 去掉停牌的股票
    # g.holding_list_W=[stock for stock in g.holding_list_W if stock in g.feasible_stocks]
    stocks_wait_spilt_d = get_stocks(g.feasible_stocks,
                                     context,
                                     g.factors,
                                     asc=g.sort_rank,
                                     factor_name=g.factor,
                                     num=2)
    stockWaitSell = []
    if len(stocks_wait_spilt_d) != 0:
        holding_list_D = [stock for stock in stocks_wait_spilt_d if stock in g.holding_list_W]
    if len(list(context.portfolio.positions.keys())) != 0:
        stockWaitSell = [stock for stock in holding_list_D if stock in list(context.portfolio.positions.keys())][
                        -int(g.stock_per_day * g.num_stocks_D) - 1:-1]
        print
        '检查百分比调仓：', len(holding_list_D), len(stockWaitSell), len(list(context.portfolio.positions.keys()))

    # 9 重新调整仓位，输入context,使用信号结果holding_list
    if True:  # 原本用来判断是否需要先跑一段数据再执行
        if g.list_count_set[0] > g.periods:
            rebalance(context, holding_list_D, stockWaitSell)
        else:
            print
            '获取数据阶段，不进行调仓'
    else:
        rebalance(context, holding_list_D, stockWaitSell)
        g.if_trade = False


# 7 获得因子信息
# stocks_list调用g.feasible_stocks factors调用字典g.factors
# 输出所有对应数据和对应排名，DataFrame
def get_factors(stocks_list, context, factors, num):
    # 从可行股票池中生成股票代码列表
    df_all_raw = pd.DataFrame(stocks_list)
    # 修改index为股票代码
    df_all_raw['code'] = df_all_raw[0]
    df_all_raw.index = df_all_raw['code']
    # 格式调整，没有一步到位中间有些东西还在摸索，简洁和效率的一个权衡
    del df_all_raw[0]
    stocks_list300 = list(df_all_raw.index)
    # 判断是否为多因子
    # if g.factor == 'AllFactor':
    tmp = 'get_df' + '_' + 'AllFactor'
    value = g.factors['AllFactor']
    # 声明字符串是个方程
    aa = globals()[tmp](stocks_list, context, value, num)
    if type(aa) != bool:
        """    
        else:
        # 每一个指标量都合并到一个dataframe里
            for key,value in g.factors.items():
                # 构建一个新的字符串，名字叫做 'get_df_'+ 'key'
                tmp='get_df' + '_' + key
                # 声明字符串是个方程
                aa = globals()[tmp](stocks_list, context, value)
                # 合并处理
                df_all_raw = aa
        """
        # 删除code列
        df_all_raw = aa
        # del df_all_raw['code']
        # 对于新生成的股票代码取list
        stocks_list_more = list(df_all_raw.index)
        # 可能在计算过程中并如的股票剔除
        for stock in stocks_list_more[:]:
            if stock not in stocks_list300:
                df_all_raw.drop(stock)
        return df_all_raw
    else:
        return False


        # 8获得调仓信号


# 原始数据重提取因子打分排名
def get_stocks(stocks_list, context, factors, asc, factor_name, num):
    # 7获取原始数据
    df_all_raw1 = get_factors(stocks_list, context, factors, num)
    # 根据factor生成列名
    if type(df_all_raw1) != bool:
        score = factor_name + '_' + 'sorted_rank'
        stocks = list(df_all_raw1.sort(score, ascending=asc).index)
        # print '周期',num,len(stocks)
        return stocks
    else:
        return []


# 9交易调仓
# 依本策略的买入信号，得到应该买的股票列表
# 借用买入信号结果，不需额外输入
# 输入：context（见API）
def rebalance(context, holding_list, stockWaitSell):
    # 每只股票购买金额
    print
    '待选股票池数量：', len(holding_list)
    every_stock = context.portfolio.portfolio_value / g.num_stocks_D
    # g.stock_per_day  g.num_stocks_D
    # 空仓只有买入操作
    if len(list(context.portfolio.positions.keys())) == 0:
        # 原设定重scort始于回报率相关打分计算，回报率是升序排列
        for stock_to_buy in list(holding_list)[0:g.num_stocks_D]:
            order_target_value(stock_to_buy, every_stock)
    else:
        # 不是空仓先卖出持有但是不在购买名单中而且可以交易的股票
        for stock_to_sell in stockWaitSell:
            if stock_to_sell in g.feasible_stocks:
                order_target_value(stock_to_sell, 0)
        # 因order函数调整为顺序调整，为防止先行调仓股票由于后行调仓股票占金额过大不能一次调整到位，这里运行两次以解决这个问题
        stockWaitBuy = [stock for stock in holding_list if stock not in list(context.portfolio.positions.keys())][
                       :int(g.stock_per_day * g.num_stocks_D)]
        for stock_to_buy in stockWaitBuy:
            order_target_value(stock_to_buy, every_stock)
        for stock_to_buy in stockWaitBuy:
            order_target_value(stock_to_buy, every_stock)
    print
    '当前持仓股票数量:', len(context.portfolio.positions.keys())


# 4
# 某一日的前shift个交易日日期
# 输入：date为datetime.date对象(是一个date，而不是datetime)；shift为int类型
# 输出：datetime.date对象(是一个date，而不是datetime)
def shift_trading_day(date, shift):
    # 获取所有的交易日，返回一个包含所有交易日的 list,元素值为 datetime.date 类型.
    tradingday = get_all_trade_days()
    # 得到date之后shift天那一天在列表中的行标号 返回一个数
    shiftday_index = list(tradingday).index(date) + shift
    # 根据行号返回该日日期 为datetime.date类型
    return tradingday[shiftday_index]


# 行业中性化处理，按照证监会标准分为十八个行业，选择股票时从相应行业选取相应比例的股票，
# 如果某一行业股票过少不足一支时取一支,输入一个series结构index为股票代码，内容为选股因子,再输入因子名称格式为字符串
def INDUSTRY_SORTED(stock_list, df_FACTOR, FACTOR_NAME):
    # 获得行业信息
    # 将stock_list的股票代码转换为国泰安的格式
    def stock_code_trun(stock_list):
        gta_stock_list = []
        # stock_list=list(stock_list)
        for stock in stock_list:
            temp_stock = stock[:6]
            gta_stock_list.append(temp_stock)
        return gta_stock_list

    gta_stock_list = stock_code_trun(stock_list)
    # 获取行业信息,一次最多返回3000条信息，因此要筛掉无用信息
    df_stk = gta.run_query(query(gta.STK_INSTITUTIONINFO.SYMBOL, gta.STK_INSTITUTIONINFO.INDUSTRYCODE
                                 ).filter(gta.STK_INSTITUTIONINFO.SYMBOL.in_(gta_stock_list)))
    # 得到一个series存放股票所在行业
    series_industry = df_stk.ix[:, 'INDUSTRYCODE']
    # 将股票代码与行业信息对应
    # 将股票代码转化为聚宽的格式
    list_code_temp = list()
    for symbol_temp in df_stk.ix[:, 'SYMBOL']:
        list_code_temp.append(normalize_code(symbol_temp))
    series_industry.index = list_code_temp
    # 去掉不需要的行业信息
    temp = list(df_FACTOR.index)
    series_industry = series_industry[temp]
    # 丢掉没有行业信息的数据
    series_industry = series_industry.dropna()
    # 将相应的缺失行业的因子信息一并删掉
    temp = list(series_industry.index)
    df_FACTOR = df_FACTOR.ix[temp]
    # 去掉行业编码只留下行业类别，例如A01去掉01留下A
    list_INDUSTRY_temp = list()
    for industrycode_temp in series_industry:
        list_INDUSTRY_temp.append(industrycode_temp[0])
    # 将股票代码、factor、行业信息写入一个DataFrame
    df_temp_p = DataFrame(columns=[FACTOR_NAME, 'INDUSTRY'])
    df_temp_p[FACTOR_NAME] = df_FACTOR[FACTOR_NAME]
    df_temp_p['INDUSTRY'] = list_INDUSTRY_temp
    # 删除空值
    df_temp_p = df_temp_p.dropna()
    # 行业中性处理，防止股票集中在个别行业，取每个行业的股票不超过总股票数的g.precent支
    # 列出所有行业分类
    list_INDUSTRY = list('ABCDEFGHIJKLMNPQRS')
    # 筛选行业
    df_INDUSTRY_selected = DataFrame(columns=[FACTOR_NAME, 'INDUSTRY'])
    # 本程序里排序rank和sort不一样rank=True时sort=False排序才一样
    g.precent = g.stock_num_M / len(df_temp_p)
    sort_asc = not g.sort_rank
    for industry in list_INDUSTRY:
        df_temp = df_temp_p[df_temp_p.INDUSTRY == industry]
        if len(df_temp) > 0:
            df_temp = df_temp.sort(columns=FACTOR_NAME, ascending=sort_asc)
            # 选取该行业的g.percent的股票，其余股票删掉
            num_temp = len(df_temp) * g.precent
            # 取整再+1
            num_temp = int(num_temp) + 1
            df_temp = df_temp[:num_temp]
            # 添加到股票池
            df_INDUSTRY_selected = pd.concat([df_INDUSTRY_selected, df_temp])
        else:
            pass
    # 有些公司可能会分到不同的行业，需要去重
    df_INDUSTRY_selected['temp_index'] = df_INDUSTRY_selected.index
    df_INDUSTRY_selected = df_INDUSTRY_selected.drop_duplicates(subset='temp_index')
    # 删除多余信息
    del df_INDUSTRY_selected['INDUSTRY']
    del df_INDUSTRY_selected['temp_index']
    return df_INDUSTRY_selected


# 一天的涨跌幅（收益率）
def get_df_Price_change_D(stock_list, context, asc):
    # 上个交易日日期
    yest = context.previous_date
    # 一个shift前的交易日日期
    days_1shift_before = shift_trading_day(yest, shift=-1)
    # 获得上个交易日收盘价
    df_price_info = get_price(list(stock_list),
                              start_date=yest,
                              end_date=yest,
                              frequency='daily',
                              fields='close')['close'].T
    # 1个交易日前收盘价信息
    df_price_info_1shift = get_price(list(stock_list),
                                     start_date=days_1shift_before,
                                     end_date=days_1shift_before,
                                     frequency='daily',
                                     fields='close')['close'].T
    # 1天的收益率,Series
    Series_mtd1 = (df_price_info.ix[:, yest]
                   - df_price_info_1shift.ix[:, days_1shift_before]
                   ) / df_price_info_1shift.ix[:, days_1shift_before]
    # 生成DataFrame
    df_Price_change_D = DataFrame()
    df_Price_change_D['Price_change_D'] = Series_mtd1
    # 排序给出排序打分，MTD1
    df_Price_change_D['Price_change_D_sorted_rank'] = df_Price_change_D['Price_change_D'].rank(ascending=asc,
                                                                                               method='dense')
    return df_Price_change_D


# 一周的涨跌幅（收益率）
def get_df_Price_change_W(stock_list, context, asc):
    # 上个交易日日期
    yest = context.previous_date
    # 一个shift前的交易日日期
    days_1shift_before = shift_trading_day(yest, shift=-5)
    # 获得上个交易日收盘价
    df_price_info = get_price(list(stock_list),
                              start_date=yest,
                              end_date=yest,
                              frequency='daily',
                              fields='close')['close'].T
    # 1个交易日前收盘价信息
    df_price_info_1shift = get_price(list(stock_list),
                                     start_date=days_1shift_before,
                                     end_date=days_1shift_before,
                                     frequency='daily',
                                     fields='close')['close'].T
    # 1周的收益率,Series
    Series_mtw1 = (df_price_info.ix[:, yest]
                   - df_price_info_1shift.ix[:, days_1shift_before]
                   ) / df_price_info_1shift.ix[:, days_1shift_before]
    # 生成DataFrame
    df_Price_change_W = DataFrame()
    df_Price_change_W['Price_change_W'] = Series_mtw1
    # 排序给出排序打分，MTD1
    df_Price_change_W['Price_change_W_sorted_rank'] = df_Price_change_W['Price_change_W'].rank(ascending=asc,
                                                                                               method='dense')
    return df_Price_change_W


# 一个月收益（涨跌幅）
def get_df_Price_change_M(stock_list, context, asc):
    # 上个交易日日期
    yest = context.previous_date
    # 一个shift前的交易日日期
    days_1shift_before = shift_trading_day(yest, shift=-21)
    # 获得上个交易日收盘价
    df_price_info = get_price(list(stock_list),
                              start_date=yest,
                              end_date=yest,
                              frequency='daily',
                              fields='close')['close'].T
    # 1个月前收盘价信息
    df_price_info_1shift = get_price(list(stock_list),
                                     start_date=days_1shift_before,
                                     end_date=days_1shift_before,
                                     frequency='daily',
                                     fields='close')['close'].T
    # 1月的收益率,Series
    Series_mtm1 = (df_price_info.ix[:, yest]
                   - df_price_info_1shift.ix[:, days_1shift_before]
                   ) / df_price_info_1shift.ix[:, days_1shift_before]
    # 生成DataFrame
    df_Price_change_M = DataFrame()
    df_Price_change_M['Price_change_M'] = Series_mtm1
    # 排序给出排序打分，MTM1
    df_Price_change_M['Price_change_M_sorted_rank'] = df_Price_change_M['Price_change_M'].rank(ascending=asc,
                                                                                               method='dense')
    return df_Price_change_M


# 1 账面市值比
def get_df_BP(stock_list, context, asc):
    df_BP = get_fundamentals(query(valuation.code, valuation.pb_ratio
                                   ).filter(valuation.code.in_(stock_list)))
    # 获得pb倒数
    df_BP['BP'] = df_BP['pb_ratio'].apply(lambda x: 1 / x)
    df_BP.index = df_BP['code']
    # 删除nan
    df_BP = df_BP.dropna()
    df_BP['BP_sorted_rank'] = df_BP['BP'].rank(ascending=asc, method='dense')
    return df_BP


# 2 一个月动能，输入stock_list, context, asc = True/False
# 输出：dataframe，index为code
def get_df_MTM1(stock_list, context, asc):
    # 上个交易日日期
    yest = context.previous_date
    # 一个shift前的交易日日期
    days_1shift_before = shift_trading_day(yest, shift=-21)
    # 获得上个交易日收盘价
    df_price_info = get_price(list(stock_list),
                              start_date=yest,
                              end_date=yest,
                              frequency='daily',
                              fields='close')['close'].T
    # 1个月前收盘价信息
    df_price_info_1shift = get_price(list(stock_list),
                                     start_date=days_1shift_before,
                                     end_date=days_1shift_before,
                                     frequency='daily',
                                     fields='close')['close'].T
    # 1月的收益率,Series
    Series_mtm1 = (df_price_info.ix[:, yest]
                   - df_price_info_1shift.ix[:, days_1shift_before]
                   ) / df_price_info_1shift.ix[:, days_1shift_before]
    # 生成DataFrame
    df_MTM1 = DataFrame()
    df_MTM1['MTM1'] = Series_mtm1
    # 排序给出排序打分，MTM1
    df_MTM1['MTM1_sorted_rank'] = df_MTM1['MTM1'].rank(ascending=asc, method='dense')
    return df_MTM1


# 3 三个月动能，输入stock_list, context, asc = True/False
# 输出：dataframe，index为code
def get_df_MTM3(stock_list, context, asc):
    # 上个交易日日期
    yest = context.previous_date
    # 3个shift前的交易日日期
    days_3shift_before = shift_trading_day(yest, shift=-63)
    # 获得上个交易日收盘价
    df_price_info = get_price(list(stock_list),
                              start_date=yest,
                              end_date=yest,
                              frequency='daily',
                              fields='close')['close'].T
    # 3个月前收盘价信息
    df_price_info_3shift = get_price(list(stock_list),
                                     start_date=days_3shift_before,
                                     end_date=days_3shift_before,
                                     frequency='daily',
                                     fields='close')['close'].T
    # 3个月的收益率,Series
    Series_mtm3 = (df_price_info.ix[:, yest]
                   - df_price_info_3shift.ix[:, days_3shift_before]
                   ) / df_price_info_3shift.ix[:, days_3shift_before]
    # 生成DataFrame
    df_MTM3 = DataFrame()
    df_MTM3['MTM3'] = Series_mtm3
    # 排序给出排序打分，MTM3
    df_MTM3['MTM3_sorted_rank'] = df_MTM3['MTM3'].rank(ascending=asc, method='dense')
    return df_MTM3


# 4 PPReversal 东方证券“乒乓球反转因子” PPReversal=5日均价/60日成交均价
# 一个月动能，输入stock_list, context, asc = True/False
# 输出：dataframe，index为code
def get_df_PPReversal(stock_list, context, asc):
    # 上个交易日日期
    yest = context.previous_date
    # 交易日日期
    date_5days_before = shift_trading_day(yest, shift=-5)
    date_60days_before = shift_trading_day(yest, shift=-60)
    # 获得5日均价
    df_price_5 = get_price(list(stock_list),
                           start_date=date_5days_before,
                           end_date=yest,
                           frequency='daily',
                           fields='avg').mean()
    # 60日均价
    df_price_60 = get_price(list(stock_list),
                            start_date=date_60days_before,
                            end_date=yest,
                            frequency='daily',
                            fields='avg').mean()
    # PPReversal
    df_PPReversal = df_price_5 / df_price_60
    df_PPReversal.columns = ['PPReversal']
    df_PPReversal = df_PPReversal.dropna()
    # 排序给出排序打分，MTM1
    df_PPReversal['PPReversal_sorted_rank'] = df_PPReversal['PPReversal'].rank(ascending=asc, method='dense')
    return df_PPReversal


# 5 CGO Capital Gains Overhang
# Rt+l=(Vt)(Pt) + (1-Vt)(Rt) # R考虑过去一段时间的换手率加权的平均价格 成交量越高的成交价格权重越大
# where
# t = a trading day
# V = the daily turnover
# P = the stock price
# R = the reference price
# CGO =（P(t-5)-R(t-5)）/P(t-5) # (t-5)为下标表示时间
# CGO表示现在价格与之前一段时间成交量加权平均价格的偏差
# 这里R用前三个月的的换手率加权平均值
def get_df_CGO(stock_list, context, asc):
    # 上个交易日日期
    yest = context.previous_date
    # 5天前
    date_5days_before = shift_trading_day(yest, shift=-5)
    str_5days_before = str(date_5days_before)
    # 3个shift前的交易日日期
    date_3shift_before = shift_trading_day(yest, shift=-63)
    # 获取过去三个月每个交易日，聚宽财务数据api只能获取某一天的财务数据
    list_date = []
    for i in range(63, 4, -1):
        temp_date = shift_trading_day(yest, shift=-i)
        list_date.append(temp_date)
    # 3个月前至5天前的均价
    panel_price_3shift = get_price(list(stock_list),
                                   start_date=date_3shift_before,
                                   end_date=date_5days_before,
                                   frequency='daily',
                                   fields=['avg', 'volume'])

    # 获取三个月至五天前换手率
    df_TR = DataFrame()
    for date in list_date:
        df_temp = get_fundamentals(query(valuation.code, valuation.turnover_ratio
                                         ).filter(valuation.code.in_(stock_list)), date)
        df_temp.index = df_temp.code
        df_TR = pd.concat([df_TR, df_temp.turnover_ratio], axis=1)
    # 取转置
    df_TR = df_TR.T

    # 计算R(t-5)值
    def get_df_R(df_price, df_TR):
        L = len(df_price)
        df_R = DataFrame(columns=df_price.columns)
        df_temp = df_price.ix[0, :]
        df_R = df_R.append(df_temp)
        for i in range(1, L):
            df_temp = df_TR.ix[i, :] * df_price.ix[i, :] + (1 - df_TR.ix[i, :]) * df_R.ix[i - 1, :]
            df_R = df_R.append(df_temp, ignore_index=True)
        return df_R

    df_price = panel_price_3shift['avg']
    df_R = get_df_R(df_price, df_TR)
    # 取前5天那一天的值
    temp_L = len(panel_price_3shift['volume'])
    df_R = df_R.ix[temp_L - 1, :]
    df_price = df_price.ix[temp_L - 1, :]
    # CGO值
    series_CGO = (df_price - df_R) / df_price
    # 生成DataFrame
    df_CGO = DataFrame(series_CGO)
    df_CGO.columns = ['CGO']
    # 删除NaN
    df_CGO = df_CGO.dropna()
    # 排序给出排序打分
    df_CGO['CGO_sorted_rank'] = df_CGO['CGO'].rank(ascending=asc, method='dense')
    return df_CGO


# 6 TO 流通股本日均换手率
def get_df_TO(stock_list, context, asc):
    # 获取价格数据,当前到21天前一共22行，与之前get_price不同，没有使用转置，行为股票代码
    # 获得换手率(%)turnover_ratio
    df_TO = get_fundamentals(query(valuation.code, valuation.turnover_ratio
                                   ).filter(valuation.code.in_(stock_list)), context.previous_date)
    # # 删除nan
    # 使用股票代码作为index
    df_TO.index = df_TO.code
    # 删除无用数据
    del df_TO['code']
    # 删除NaN
    df_TO = df_TO.dropna()
    # 换名字
    df_TO.columns = ['TO']
    # 生成排名序数
    df_TO['TO_sorted_rank'] = df_TO['TO'].rank(ascending=asc, method='dense')
    return df_TO


# 7 EPS_GR(EPS Growth Ratio)每股收益增长率（年度同比）
def get_df_EPS_GR(stock_list, context, asc):
    # 获取日期
    yest = context.previous_date
    date_oneyear_before = shift_trading_day(yest, shift=-243)
    # 查询财务数据
    df_EPS = get_fundamentals(query(indicator.code, indicator.eps
                                    ).filter(indicator.code.in_(stock_list)), date=yest)
    df_EPS.index = df_EPS.code
    series_EPS = df_EPS['eps']
    df_EPS_oneyear_before = get_fundamentals(query(indicator.code, indicator.eps
                                                   ).filter(indicator.code.in_(stock_list)), date=date_oneyear_before)
    df_EPS_oneyear_before.index = df_EPS_oneyear_before.code
    series_EPS_oneyear_before = df_EPS_oneyear_before['eps']
    # 计算每股收益增长率DPS_GR
    df_EPS_GR = DataFrame(columns=['EPS_GR'])
    df_EPS_GR['EPS_GR'] = (series_EPS - series_EPS_oneyear_before) / series_EPS
    # 删除NaN
    df_EPS_GR = df_EPS_GR.dropna()
    # 生成排名序数
    df_EPS_GR['EPS_GR_sorted_rank'] = df_EPS_GR['EPS_GR'].rank(ascending=asc, method='dense')
    return df_EPS_GR


# 8 Revenue Growth Ratio营业收入增长率 季度同比
def get_df_RGR(stock_list, context, asc):
    # 获取日期
    yest = context.previous_date
    control_date = 0
    date_num = -63
    while 1:
        date_oneseason_before = shift_trading_day(yest, shift=date_num)
        # 查询财务数据
        df_Revenue = get_fundamentals(query(income.code, income.operating_revenue
                                            ).filter(income.code.in_(stock_list)), date=yest)
        df_Revenue.index = df_Revenue.code
        series_Revenue = df_Revenue['operating_revenue']
        df_Revenue_oneseason_before = get_fundamentals(query(income.code, income.operating_revenue
                                                             ).filter(income.code.in_(stock_list)),
                                                       date=date_oneseason_before)
        df_Revenue_oneseason_before.index = df_Revenue_oneseason_before.code
        series_Revenue_oneseason_before = df_Revenue_oneseason_before['operating_revenue']
        # 计算营业收入增长率
        df_RGR = DataFrame(columns=['RGR'])
        df_RGR['RGR'] = (series_Revenue - series_Revenue_oneseason_before) / series_Revenue
        # 删除NaN
        df_RGR = df_RGR.dropna()
        # 生成排名序数
        df_RGR['RGR_sorted_rank'] = df_RGR['RGR'].rank(ascending=asc, method='dense')
        control_date = len([i for i in list(df_RGR['RGR']) if i > 0.0 or i < 0.0])
        date_num -= 10
        if control_date > 0.9 * len(list(df_RGR['RGR'])):
            return df_RGR


# 9 净利润增长率季度同比 net_profit Growth Ratio
def get_df_NPGR(stock_list, context, asc):
    yest = context.previous_date
    control_date = 0
    date_num = -63
    while 1:
        date_oneseason_before = shift_trading_day(yest, shift=date_num)
        # 查询财务数据
        df_NP = get_fundamentals(query(income.code, income.net_profit
                                       ).filter(income.code.in_(stock_list)), date=yest)
        df_NP.index = df_NP.code
        series_NP = df_NP['net_profit']
        df_NP_oneseason_before = get_fundamentals(query(income.code, income.net_profit
                                                        ).filter(income.code.in_(stock_list)),
                                                  date=date_oneseason_before)
        df_NP_oneseason_before.index = df_NP_oneseason_before.code
        series_NP_oneseason_before = df_NP_oneseason_before['net_profit']
        # 计算净利润增长率
        df_NPGR = DataFrame(columns=['NPGR'])
        df_NPGR['NPGR'] = (series_NP - series_NP_oneseason_before) / series_NP
        # 删除NaN
        df_NPGR = df_NPGR.dropna()
        # 生成排名序数
        df_NPGR['NPGR_sorted_rank'] = df_NPGR['NPGR'].rank(ascending=asc, method='dense')
        control_date = len([i for i in list(df_NPGR['NPGR']) if i > 0.0 or i < 0.0])
        date_num -= 10
        if control_date > 0.9 * len(list(df_NPGR['NPGR'])):
            return df_NPGR


# 10 非流动性因子illiquidity factor(ILLIQ)
# 市场的流动性越差价格让步越大 越能获得超额收益 但是这个因子受小市值影响明显
# ILLIQ=(1/N)sum(abs(Ri)/Vi) 每日价格变化幅度绝对值和成交额的比值求平均 这里N取5，即过去5天求平均
def get_df_ILLIQ(stock_list, context, asc):
    yest = context.previous_date
    date_5days_before = shift_trading_day(yest, shift=-4)  # 包含yest以及之前的4天数据共5天
    date_6days_before = shift_trading_day(yest, shift=-5)
    # 获取涨跌幅信息 避免时间差引起的股票不一致获取六天数据 去掉最新的一天
    df_volume = get_price(list(stock_list),
                          start_date=date_6days_before,
                          end_date=yest,
                          frequency='daily',
                          fields=['close'])['close']
    df_volume = df_volume.ix[:5, :]
    # 获取成交量信息
    panel_volume = get_price(list(stock_list),
                             start_date=date_5days_before,
                             end_date=yest,
                             frequency='daily',
                             fields=['volume', 'close'])
    # 计算涨跌幅 坑爹的聚宽API不能获取一段时间的涨跌幅 只好自己计算
    df_volume.index = panel_volume['close'].index
    panel_volume['change'] = (panel_volume['close'] - df_volume) / df_volume
    # 绝对值
    panel_volume['change'] = panel_volume['change'].abs()
    temp = (panel_volume['change'] / panel_volume['volume'])
    # 成交量单位换成亿元
    panel_volume['volume'] = panel_volume['volume'] / 100000000
    df_ILLIQ = DataFrame()
    df_ILLIQ['ILLIQ'] = (panel_volume['change'] / panel_volume['volume']).sum().T * 0.2
    df_ILLIQ = df_ILLIQ.dropna()
    # 生成排名序数
    df_ILLIQ['ILLIQ_sorted_rank'] = df_ILLIQ['ILLIQ'].rank(ascending=asc, method='dense')
    return df_ILLIQ  # , series_price_change


def str_min(df_input):
    return df_input.min()


def str_max(df_input):
    return df_input.max()


# 13 收盘价与成交均价的关系
def get_df_alpha_55(stock_list, context, asc):
    # 上个交易日日期
    yest = context.previous_date
    # 一个shift前的交易日日期
    days_shift_before = shift_trading_day(yest, shift=-18)  # （6+12）
    # 获得需要的股票价格信息 按时间顺序排序最后一行为最新数据
    panel_price_info = get_price(list(stock_list),
                                 start_date=days_shift_before,
                                 end_date=yest,
                                 frequency='daily',
                                 fields=['close', 'high', 'low', 'volume'])

    # 聚宽pandas库还未更新还不支持rolling函数，自定义rolling函数(窗口平移计算)
    def rolling(df_input, window, function_name):
        if 0 < window < len(df_input):
            df_output = DataFrame(index=list(df_input.index), columns=list(df_input.columns))
            for i in range(len(df_input)):
                if i < window:
                    df_output.ix[i, :] = globals()[function_name](df_input.ix[:i + 1, :])
                else:
                    df_output.ix[i, :] = globals()[function_name](df_input.ix[i - window:i + 1, :])
        else:
            print
            'winow out of size'
        return df_output

    # 求ts_min(low,12)和ts_max(high,12)
    panel_price_info['ts_min'] = rolling(panel_price_info['low'], window=12, function_name='str_min')
    panel_price_info['ts_max'] = rolling(panel_price_info['high'], window=12, function_name='str_max')
    # 截取最新6天数据 跳过了没有数据（NaN）的交易日
    panel_price_info = panel_price_info.ix[:, -6:, :]

    # alpha中的rank函数(横截面排序 ？)
    def alpha_rank(df_input):
        temp_columns = list(df_input.columns)
        for stock in temp_columns:
            column_name = 'rank_' + stock
            df_input[column_name] = df_input[stock].rank(ascending=True, method='dense')
            df_input[column_name] = df_input[column_name] / df_input[column_name].max()
            del df_input[stock]
        df_input.columns = temp_columns
        return df_input

    # 计算rank值
    # 去掉数据不全的股票
    panel_price_info = panel_price_info.dropna(axis=2)
    # 去掉相减为零的值
    panel_price_info['max_min'] = panel_price_info['ts_max'] - panel_price_info['ts_min']
    # 将相为0的项转化为NaN
    panel_price_info['none_zero'] = panel_price_info['max_min'][panel_price_info['max_min'] != 0]
    # 去掉相关股票
    panel_price_info = panel_price_info.dropna(axis=2)
    # 计算
    panel_price_info['temp'] = (panel_price_info['close'] - panel_price_info['ts_min']) / (
    panel_price_info['ts_max'] - panel_price_info['ts_min'])
    panel_price_info['rank_a'] = alpha_rank(panel_price_info['temp'])
    panel_price_info['rank_b'] = alpha_rank(panel_price_info['volume'])
    # 计算alpha_55号因子的值 corrcoef
    # 生成DataFrame
    df_alpha_55 = DataFrame()
    # 临时列表容器
    list_temp = []
    # 计算每支股票的相关系数
    for i in range(len(panel_price_info['rank_a'].columns)):
        # corrcoef返回为矩阵取对应的相关系数
        temp_coef = corrcoef(panel_price_info['rank_a'].ix[:, i], panel_price_info['rank_b'].ix[:, i])[0, 1]
        list_temp.append(temp_coef)
    df_alpha_55['alpha_55'] = list_temp
    df_alpha_55['alpha_55'] = -df_alpha_55['alpha_55']
    df_alpha_55.index = panel_price_info['rank_a'].columns
    # 将NaN值替换为零
    df_alpha_55 = df_alpha_55.fillna(0)
    # 得到series
    series_alpha_55 = df_alpha_55['alpha_55']
    # 清空DataFrame
    df_alpha_55 = DataFrame(series_alpha_55)
    df_alpha_55.columns = ['alpha_55']
    # 排序给出排序打分
    df_alpha_55['alpha_55_sorted_rank'] = df_alpha_55['alpha_55'].rank(ascending=asc, method='dense')
    return df_alpha_55


# 11 按照Fama-French规则计算k个参数并且回归（三因子或五因子模型），计算出股票的alpha并且输出DataFrame
def get_df_FF(stock_list, context, asc):
    # 三因子NoF=3，五因子NoF=5
    NoF = 3
    # 无风险利率
    rf = 0.04
    # 时间
    yest = context.previous_date
    date_3month_before = shift_trading_day(yest, shift=-61)
    date_1year_before = shift_trading_day(yest, shift=-243)
    # 股票个数
    LoS = len(stock_list)
    # 查询三因子/五因子的语句
    q = query(
        valuation.code,
        valuation.market_cap,
        (balance.total_owner_equities / valuation.market_cap / 100000000.0).label("BTM"),
        indicator.roe,
        balance.total_assets.label("Inv")
    ).filter(valuation.code.in_(stock_list))
    df = get_fundamentals(q, yest)
    # 计算5因子再投资率的时候需要跟一年前的数据比较，所以单独取出计算
    ldf = get_fundamentals(q, date_1year_before)
    # 若前一年的数据不存在，则暂且认为Inv=0
    if len(ldf) == 0:
        ldf = df
    df["Inv"] = np.log(df["Inv"] / ldf["Inv"])
    # 选出特征股票组合
    S = df.sort('market_cap')['code'][:LoS / 3]
    B = df.sort('market_cap')['code'][LoS - LoS / 3:]
    L = df.sort('BTM')['code'][:LoS / 3]
    H = df.sort('BTM')['code'][LoS - LoS / 3:]
    W = df.sort('roe')['code'][:LoS / 3]
    R = df.sort('roe')['code'][LoS - LoS / 3:]
    C = df.sort('Inv')['code'][:LoS / 3]
    A = df.sort('Inv')['code'][LoS - LoS / 3:]
    # 获得样本期间的股票价格并计算日收益率，
    df2 = get_price(list(stock_list), date_3month_before, yest, '1d')
    df3 = df2['close'][:]
    # 取自然对数再差分求收益率（涨跌幅）的近似
    df4 = np.diff(np.log(df3), axis=0) + 0 * df3[1:]
    # 求因子的值
    SMB = sum(df4[S].T) / len(S) - sum(df4[B].T) / len(B)
    HMI = sum(df4[H].T) / len(H) - sum(df4[L].T) / len(L)
    RMW = sum(df4[R].T) / len(R) - sum(df4[W].T) / len(W)
    CMA = sum(df4[C].T) / len(C) - sum(df4[A].T) / len(A)
    # 用沪深300作为大盘基准
    dp = get_price('000001.XSHG', date_3month_before, yest, '1d')['close']
    # 取自然对数再差分求收益率（涨跌幅）的近似
    RM = diff(np.log(dp)) - rf / 243
    # 将因子们计算好并且放好
    X = pd.DataFrame({"RM": RM, "SMB": SMB, "HMI": HMI, "RMW": RMW, "CMA": CMA})
    # 取前NoF个因子为策略因子
    factor_flag = ["RM", "SMB", "HMI", "RMW", "CMA"][:NoF]
    X = X[factor_flag]

    # 线性回归函数
    def linreg(X, Y):
        X = sm.add_constant(array(X))
        Y = array(Y)
        if len(Y) > 1:
            results = regression.linear_model.OLS(Y, X).fit()
            # 这里输出
            return results.rsquared
        else:
            return [float("nan")]

    # 对样本数据进行线性回归并计算alpha
    t_scores = [0.0] * LoS
    for i in range(LoS):
        t_stock = stock_list[i]
        t_r = linreg(X, df4[t_stock] - rf / 243)
        t_scores[i] = t_r
    # 这个scores就是alpha
    df_FF = pd.DataFrame({'FF': t_scores})
    df_FF.index = stock_list
    # 去掉缺失的值
    df_FF = df_FF.dropna()
    # 生成排名序数
    df_FF['FF_sorted_rank'] = df_FF['FF'].rank(ascending=asc, method='dense')
    return df_FF


# 12 流通市值
def get_df_CMC(stock_list, context, asc):
    # 获得流通市值 circulating_market_cap 流通市值(亿)
    df_CMC = get_fundamentals(query(valuation.code, valuation.circulating_market_cap
                                    ).filter(valuation.code.in_(stock_list)))
    df_CMC.index = df_CMC['code']
    # 删除nan
    df_CMC = df_CMC.dropna()
    df_CMC['CMC'] = df_CMC['circulating_market_cap']
    # 删除无用信息
    del df_CMC['circulating_market_cap']
    del df_CMC['code']
    # 生成排名序数
    df_CMC['CMC_sorted_rank'] = df_CMC['CMC'].rank(ascending=asc, method='dense')
    return df_CMC


# 14   varc   月波动率变化
def get_df_VARC(stock_list, context, asc):
    # 一个shift前的交易日日期
    days_1shift_before = shift_trading_day(context.previous_date, shift=-21)
    # VAR当期值（上个月）获取, 排序没关系，这里随便假设
    df_price_info_between_2shift = get_price(list(stock_list),
                                             count=22,
                                             end_date=days_1shift_before,
                                             frequency='daily',
                                             fields='close')['close']
    # 生成一个空列表
    x = []
    # 计算日回报率为前一天收盘价/当天收盘价 - 1
    for i in range(0, 21):
        x.append(df_price_info_between_2shift.iloc[i + 1]
                 / df_price_info_between_2shift.iloc[i] - 1)
    # 进行转置
    df_VARC = pd.DataFrame(x).T
    # 生成方差
    df_VARC['VAR'] = df_VARC.var(axis=1, skipna=True)
    # 获取本月方差
    df_price_info_between_1shift = get_price(list(stock_list),
                                             count=22,
                                             end_date=context.previous_date,
                                             frequency='daily',
                                             fields='close')['close']
    # 生成一个空列表
    x = []
    # 计算日回报率为前一天收盘价/当天收盘价 - 1
    for i in range(0, 21):
        x.append(df_price_info_between_1shift.iloc[i + 1]
                 / df_price_info_between_1shift.iloc[i] - 1)
    # 进行转置
    df_VAR = pd.DataFrame(x).T
    # 生成方差
    df_VAR['VAR'] = df_VAR.var(axis=1, skipna=True)
    # VAR变动
    series_VARC = (df_VAR['VAR'] - df_VARC['VAR']) / df_VARC['VAR']
    # 换为DataFrame格式
    df_VARC = DataFrame(series_VARC)
    # colunms名
    df_VARC.columns = ['VARC']
    # 删除nan
    df_VARC = df_VARC.dropna()
    # 排序给出排序打分
    df_VARC['VARC_sorted_rank'] = df_VARC['VARC'].rank(ascending=asc, method='dense')
    return df_VARC


# 15   换手率变动
def get_df_TRC(stock_list, context, asc):
    # 获取过去1个月每个交易日，聚宽财务数据api只能获取某一天的财务数据
    yest = context.previous_date
    list_date = []
    for i in range(21, 0, -1):
        temp_date = shift_trading_day(yest, shift=-i)
        list_date.append(temp_date)
    # 获取过去一个月每天的换手率
    df_TR = DataFrame()
    for date in list_date:
        df_temp = get_fundamentals(query(valuation.code, valuation.turnover_ratio
                                         ).filter(valuation.code.in_(stock_list)), date)
        df_temp.index = df_temp.code
        df_TR = pd.concat([df_TR, df_temp.turnover_ratio], axis=1)
    # 求一个月换手率均值
    df_TR = DataFrame(df_TR.mean(axis=1))
    # 换手率命名
    df_TR.columns = ['TR']
    # 计算上一个月换手率
    list_date = []
    for i in range(42, 20, -1):
        temp_date = shift_trading_day(yest, shift=-i)
        list_date.append(temp_date)
    # 获取过去一个月每天的换手率
    df_TR_before = DataFrame()
    for date in list_date:
        df_temp = get_fundamentals(query(valuation.code, valuation.turnover_ratio
                                         ).filter(valuation.code.in_(stock_list)), date)
        df_temp.index = df_temp.code
        df_TR_before = pd.concat([df_TR_before, df_temp.turnover_ratio], axis=1)
    # 求一个月换手率均值
    df_TR_before = DataFrame(df_TR_before.mean(axis=1))
    # 换手率命名
    df_TR_before.columns = ['TR_before']
    # print df_TR,df_TR_before
    series_temp = df_TR['TR'] - df_TR_before['TR_before']
    series_temp = series_temp.dropna()
    # 换手率变动
    series_TRC = series_temp / df_TR_before['TR_before']
    # 格式换为DataFrame
    df_TRC = DataFrame(series_TRC)
    # columns命名
    df_TRC.columns = ['TRC']
    # 去掉空值
    df_TRC = df_TRC.dropna()
    # 生成排名序数
    df_TRC['TRC_sorted_rank'] = df_TRC['TRC'].rank(ascending=asc, method='dense')
    return df_TRC


# 16 市销率
def get_df_PS(stock_list, context, asc):
    # 获得市销率TTMps_ratio
    df_PS = get_fundamentals(query(valuation.code, valuation.ps_ratio
                                   ).filter(valuation.code.in_(stock_list)))
    df_PS.index = df_PS.code
    # 删除nan，以备数据中某项没有产生NaN
    df_PS = df_PS[pd.notnull(df_PS['ps_ratio'])]
    df_PS['PS'] = df_PS['ps_ratio']
    # 生成series
    series_PS = df_PS['PS']
    # 清空DataFrame
    df_PS = DataFrame(series_PS)
    df_PS.columns = ['PS']
    # 生成排名序数
    df_PS['PS_sorted_rank'] = df_PS['PS'].rank(ascending=asc, method='dense')
    return df_PS


# 17 股息率
def get_df_DP(stock_list, context, asc):
    # 获得dividend_payable和market_cap 应付股利(元)和总市值(亿元)
    df_DP = get_fundamentals(query(balance.code, balance.dividend_payable, valuation.market_cap
                                   ).filter(balance.code.in_(stock_list)))
    # 按公式计算
    df_DP['DP'] = df_DP['dividend_payable'] / (df_DP['market_cap'] * 100000000)
    df_DP.index = df_DP['code']
    # 删除nan
    df_DP = df_DP.dropna()
    # 生成series
    Series_DP = df_DP['DP']
    # 清空DataFrame
    df_DP = DataFrame(Series_DP)
    df_DP.columns = ['DP']
    # 生成排名序数
    df_DP['DP_sorted_rank'] = df_DP['DP'].rank(ascending=asc, method='dense')
    return df_DP


# 18 ep
def get_df_EP(stock_list, context, asc):
    df_EP = get_fundamentals(query(valuation.code, valuation.pe_ratio
                                   ).filter(valuation.code.in_(stock_list)))
    # 获得pe倒数
    df_EP['EP'] = df_EP['pe_ratio'].apply(lambda x: 1 / x)
    df_EP.index = df_EP['code']
    # 删除nan，以备数据中某项没有产生nan
    df_EP = df_EP[pd.notnull(df_EP['EP'])]
    # 生成Series
    series_EP = df_EP['EP']
    # 清空DataFrame装结果
    df_EP = DataFrame(series_EP)
    df_EP.columns = ['EP']
    # 按对应项目排序
    df_EP['EP_sorted_rank'] = df_EP['EP'].rank(ascending=asc, method='dense')
    return df_EP


# **********************多因子选股************************
# 因子去极值
def Remove_Extremum(df_column):
    # 单列DataFrame 或者Series
    # import pandas as pd
    # import numpy as np
    import math
    Te_data = df_column.values
    md = np.median(Te_data)
    md_up = []
    md_down = []
    # 计算mc
    for i in Te_data:
        if i > md:
            md_up.append(i)
        if i < md:
            md_down.append(i)
    container = []
    for i in md_up:
        for j in md_down:
            container.append(((i - md) - (md - j)) / (i - j))
    mc = np.median(np.array(container))
    # 计算L U
    Q1 = np.percentile(Te_data, 25)
    Q3 = np.percentile(Te_data, 75)
    IQR = Q3 - Q1
    if mc >= 0:
        L = Q1 - 1.5 * math.exp(-3.5 * mc) * IQR
        U = Q3 + 1.5 * math.exp(4 * mc) * IQR
    else:
        L = Q1 - 1.5 * math.exp(-4 * mc) * IQR
        U = Q3 + 1.5 * math.exp(3.5 * mc) * IQR
    # 输出结果
    Lr = len(df_column)
    DropIndex = []
    for i in range(0, Lr, 1):
        if df_column.iloc[i] > U or df_column.iloc[i] < L:
            DropIndex.append(i)
    temp = df_column[DropIndex].index
    dorp_labels = list(temp)
    df_result = df_column.drop(dorp_labels)
    return df_result


# 计算IC值
def cal_IC(df_AllFactor, price_change, num):
    # 计算每一个因子与收益的IC值
    # df_AllFactor为DataFrame每一个columns为对应因子名称
    # price_change为DataFrame或者series为股票下一期的收益率
    # 计算排名序数
    df_AllFactor = df_AllFactor.dropna()
    # 生成装排名的DataFrame
    df_AllFactor_rank = 0.0 * df_AllFactor
    for column in df_AllFactor.columns:
        df_AllFactor_rank[column] = df_AllFactor[column].rank(ascending=False, method='dense')
    # 添加涨跌幅排名
    price_change = price_change.dropna()
    temp_name = g.list_factor_all[num][-1]
    df_AllFactor_rank[temp_name] = price_change.rank(ascending=False, method='dense')
    # 计算IC值
    df_AllFactor_rank = df_AllFactor_rank.dropna()
    list_IC = []
    for column in list(df_AllFactor.columns):
        temp_IC = np.corrcoef(df_AllFactor_rank[column], df_AllFactor_rank[temp_name])[0, 1]
        list_IC.append(temp_IC)
    df_IC = DataFrame(data=list_IC, index=list(df_AllFactor.columns))
    return df_IC


# 计算因子值并将因子值写入全局变量
def factor_cal(stock_list, context, asc, num):
    # 时间
    yest = context.previous_date
    # 获取去极值后的因子值
    for name in g.list_factor_all[num]:
        temp = 'get_df_' + name
        # 获取因子值
        temp_series = globals()[temp](stock_list, context, asc)[name]
        # 去极值
        if g.Drop_Extremum:
            temp_series = Remove_Extremum(temp_series)
        # 结果存入series变量
        globals()[name] = temp_series
    # 因子汇总 并保存到全局变量
    # 将因子所含股票代码取并集
    set_temp = set(globals()[g.list_factor_all[num][0]].index)
    for i in range(len(g.list_factor_all[num])):
        if i > 0:
            set_temp = set_temp | set(globals()[g.list_factor_all[num][i]].index)
        else:
            pass
    list_temp_index = list(set_temp)
    # ******未标准化因子存储*******
    # 定义DataFrame作为这一期所有因子以及涨跌幅的容器（去极值后的因子值，未标准化）
    df_AllFactor_value = DataFrame(index=list_temp_index)
    # 将因子值写入DataFrame
    for columns_name in g.list_factor_all[num]:
        df_AllFactor_value[columns_name] = globals()[columns_name]
        # print df_AllFactor_value[columns_name]
    # 写入全局变量
    str_date = str(yest)
    panel_temp = pd.Panel({str_date: df_AllFactor_value})

    # g.list_panel_set[num]=pd.concat([g.list_panel_set[num],panel_temp])
    # ******标准化后因子存储**********
    # 数据标准化函数
    def Z_ScoreNormalization(x):
        # import numpy as np
        x = x.dropna()
        x = (x - np.average(x)) / np.std(x)
        return x

    if g.normalization:
        # 生成DataFrame（去极值、标准化后的因子值）
        df_AllFactor = DataFrame(index=list_temp_index)
        # 将因子值标准化并写入DataFrame
        for columns_name in g.list_factor_all[num]:
            temp = globals()[columns_name]
            temp = Z_ScoreNormalization(temp)
            df_AllFactor[columns_name] = temp
        # 写入全局变量
        str_date = str(yest)
        panel_temp = pd.Panel({str_date: df_AllFactor})

        g.list_panel_set[num] = pd.concat([g.list_panel_set[num], panel_temp])
    else:
        g.list_panel_set[num] = pd.concat([g.list_panel_set[num], panel_temp])


# 线性回归选股函数     这里系数需要确定
def linear_regression_choose(stock_list, context, asc, num):
    # *******************
    # 因子线性回归系数加权
    df_AllFactor_value = g.list_panel_set[num].ix[-1, :, :-1].dropna()
    # 因子值按线性回归结果加权
    # 各因子权重
    series_linear_weight = Series([-0.130625,
                                   -0.2776,
                                   -0.895675,
                                   2.237025,
                                   -0.4901,
                                   -5.46015,
                                   0.3308,
                                   -0.1371,
                                   0.07575,
                                   0.1363,
                                   -0.635675,
                                   0.002425])
    series_linear_weight.index = list(df_AllFactor_value.columns)
    # 加权后的因子值
    df_AllFactor_value = df_AllFactor_value * series_linear_weight
    # 因子值求和 这里格式变为Series
    df_AllFactor_value = df_AllFactor_value.sum(axis=1)
    # 格式变为DataFrame并命名因子
    df_AllFactor = DataFrame(df_AllFactor_value)
    df_AllFactor.columns = ['AllFactor']
    # 去掉空值
    df_AllFactor = df_AllFactor.dropna()
    # 行业中性化
    if g.Industry_Neutral:
        df_AllFactor = INDUSTRY_SORTED(stock_list, df_AllFactor, 'AllFactor')
    # 求排名序数
    df_AllFactor['AllFactor_sorted_rank'] = df_AllFactor['AllFactor'].rank(ascending=asc, method='dense')
    return df_AllFactor
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # 线性回归得到因子权重
    if g.regression_method:
        # 线性回归得到回归系数
        if g.list_count_set[0] > g.periods:
            # 线性回归取得因子权重
            # 取一个periods收益率最高的1/3股票回归因子系数
            series_cumulate_profit = g.list_panel_set[num].ix[1:, :, -1].sum(axis=1)
            # 收益排序
            series_cumulate_profit = series_cumulate_profit.sort(ascending=False, inplace=False)
            # 取收益最高1/3股票
            series_selected = series_cumulate_profit[:len(series_cumulate_profit) // 3]
            # 每一个因子均取收益最高1/3股票
            g.list_panel_set[num] = g.list_panel_set[num].ix[:, series_selected.index, :]
            # **************
            # 取得因子值的均值 不要‘price_change’，格式：columns为因子名称，index为时期
            df_factor_mean = g.list_panel_set[num].ix[:-1, :, :-1].mean().T
            # 取得收益‘price_change’
            series_price_mean = g.list_panel_set[num].ix[1:, :, -1].mean()
            # 调整index
            series_price_mean.index = df_factor_mean.index
            # 线性回归
            X = sm.add_constant(df_factor_mean)
            results = regression.linear_model.OLS(series_price_mean, X).fit()
            # 因子的回归系数
            print
            results.summary()
            # else:
            # 这里随便取因子，以免报错，不会实际交易
            df_AllFactor = DataFrame({'AllFactor': MTM1})
            df_AllFactor['AllFactor_sorted_rank'] = df_AllFactor['AllFactor'].rank(ascending=asc, method='dense')
            df_AllFactor.columns = ['AllFactor', 'AllFactor_sorted_rank']
        else:
            # 这里随便取因子，以免报错，不会实际交易
            df_AllFactor = DataFrame({'AllFactor': MTM1})
            df_AllFactor['AllFactor_sorted_rank'] = df_AllFactor['AllFactor'].rank(ascending=asc, method='dense')
            df_AllFactor.columns = ['AllFactor', 'AllFactor_sorted_rank']
            return df_AllFactor


# 最优化选股函数
def optimal_choose(stock_list, context, asc, num):
    # 最优化选股
    # *******************
    # 计数
    temp_count = len(g.list_panel_set[num].items)
    # 计算Rank IC
    if temp_count > 1:
        # 因子值
        df_all_factor = g.list_panel_set[num].ix[-2, :, :-1]
        # 涨跌幅
        series_price_change = g.list_panel_set[num].ix[-1, :, -1]
        # 时间
        yest = context.previous_date
        str_date = str(yest)
        g.list_df_IC_set[num][str_date] = cal_IC(df_all_factor, series_price_change, num)
        # 计算相关系数矩阵 g.periods
        temp_count = len(g.list_df_IC_set[num].T)
        if temp_count > g.period_all[num] - 1:
            df_temp = g.list_df_IC_set[num].ix[:, -g.period_all[num]:]
            df_temp = df_temp.dropna()
            array_IC_mean = array(df_temp.mean(axis=1))
            array_corr = np.corrcoef(df_temp)
    # 最优化 计算权重向量
    import scipy.optimize as sco
    def statistics(weights):
        weights = np.array(weights)
        ic_mean = array_IC_mean
        ic_cov = array_corr
        port_ir = np.dot(weights.T, ic_mean) / np.sqrt(np.dot(weights.T, np.dot(ic_cov, weights)))
        return port_ir

    def min_ir(weights):
        return -statistics(weights)

    def port_weight():
        noa = len(g.list_factor_all[num][:-1])
        # 约束是所有参数(权重)的总和为1。这可以用minimize函数的约定表达如下
        # cons = ({'type':'eq', 'fun':lambda x: np.sum(x)-1})
        # 我们还将参数值(权重)限制在0和1之间。这些值以多个元组组成的一个元组形式提供给最小化函数
        bnds = tuple((-1, 1) for x in range(noa))
        # 优化函数调用中忽略的唯一输入是起始参数列表(对权重的初始猜测)。我们简单的使用平均分布。, constraints = cons
        optv = sco.minimize(min_ir, noa * [1. / noa, ], method='SLSQP', bounds=bnds)
        # 返回权重向量
        return optv['x'].round(3)

    if g.list_count_set[0] > g.period_all[num]:
        weight_vector = port_weight()
        # 生成series
        series_weight = Series(data=weight_vector, index=g.list_factor_all[num][:-1])
        df_temp = g.list_panel_set[num].ix[-1, :, :-1].dropna()
        # 加权后的因子值
        df_temp = df_temp * series_weight
        # 清空DataFrame
        df_AllFactor = DataFrame()
        df_AllFactor['AllFactor'] = df_temp.T.sum()
        # 行业中性化
        if g.Industry_Neutral:
            df_AllFactor = INDUSTRY_SORTED(stock_list, df_AllFactor, 'AllFactor')
        df_AllFactor['AllFactor_sorted_rank'] = df_AllFactor['AllFactor'].rank(ascending=asc, method='dense')
        # print 'opt:',len(df_AllFactor)
        return df_AllFactor
    else:
        # 这里随便取因子，以免报错，不会实际交易
        df_AllFactor = DataFrame({'AllFactor': MTM1})
        df_AllFactor['AllFactor_sorted_rank'] = df_AllFactor['AllFactor'].rank(ascending=asc, method='dense')
        df_AllFactor.columns = ['AllFactor', 'AllFactor_sorted_rank']
        return df_AllFactor


# 准备训练数据
def excute_xy(num):
    # 这里需要尝试把数据存起来，回测速度太慢
    y = []
    x = []
    # 清洗X,Y
    if g.list_count_set[num] > 1:
        dataframe_old_panel = g.list_panel_set[num][-2, :, :]
        dataframe_new_panel = g.list_panel_set[num][-1, :, :]
        print
        'before na:', len(dataframe_old_panel), len(dataframe_new_panel)
        # dataframe_old_panel.dropna(axis=0,thresh=int(len(dataframe_old_panel.columns)*0.2)
        some_pchange = pd.DataFrame()
        pchange_factor = g.p_change[num]
        some_pchange['new_price_change'] = dataframe_new_panel[pchange_factor]
        some_pchange.index = dataframe_new_panel.index
        # print some_pchange
        del dataframe_old_panel[pchange_factor]
        # 两个以上nan值则删除行，否则可以填充均值
        # for i in g.f:
        #    print i,Counter(np.isnan(np.array(list(dataframe_old_panel[i]))))
        dataframe_del_2na = dataframe_old_panel.dropna(
            thresh=int(len(dataframe_old_panel.columns) * 0.2))  # int(len(dataframe_old_panel.columns)*0.6）
        print
        '数据清洗前后数量对比：', len(dataframe_del_2na), len(dataframe_old_panel.dropna())
        # for s in dataframe_old_panel.columns:

        result = pd.concat([dataframe_old_panel, some_pchange.dropna()], axis=1, join_axes=[dataframe_old_panel.index])
        y.extend(list(result.dropna()['new_price_change']))
        wait_add_x = result.dropna().values
        for some_list in wait_add_x:
            x.append(list(some_list)[:-1])
        for i in x:
            g.x_all[num].append(np.array(i))
        g.y_all[num] += y
        print
        'excute xy:', len(g.x_all[num]), len(g.y_all[num])


# 训练模型

def random_forest(period_num):
    rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                max_depth=None, max_features='auto', max_leaf_nodes=None,
                                min_samples_leaf=1, min_samples_split=2,
                                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                                oob_score=False, random_state=None, verbose=0,
                                warm_start=False)
    num = int(len(g.x_all[period_num]) * g.period_all[period_num] / g.list_count_set[period_num])
    print
    str(period_num) + '周期训练数据量:' + str(num)
    x = np.array(g.x_all[period_num][-num:])
    y = np.array(g.y_all[period_num][-num:])
    bount_up = np.percentile(y, 70)  # 控制Y多分类的
    bount_down = np.percentile(y, 30)
    print
    bount_down, bount_up
    for i in range(len(y)):
        if y[i] < bount_down:
            y[i] = 0
        elif y[i] > bount_up:
            y[i] = 2
        elif y[i] >= bount_down and y[i] <= bount_up:
            y[i] = 1
    # for i in range(len(y)):
    #    y[i] = int(y[i]*10000)
    # print type(n_x),n_x
    print
    str(period_num) + '周期数据存储数量:', len(g.x_all[period_num]), len(g.y_all[period_num])
    rf.fit(x, y)
    print
    rf.score(x, y)
    names = list(g.list_panel_set[period_num][-1, :, :].columns)
    print
    sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names[:-1]),
           reverse=True)
    g.machinelearning[period_num].append(rf)


def randomtree_choose(stock_list, context, asc, num):
    excute_xy(num)
    if g.list_count_set[num] > g.period_all[num]:
        random_forest(num)
        x_na = g.list_panel_set[num][-1, :, :]
        # print x.index
        # predict_record = []
        stocks_l = []
        del x_na[g.p_change[num]]
        x = x_na.dropna()
        n_x = []
        for i in range(len(x)):
            n_x.append(np.array(x.iloc[i]))
        n_x = np.array(n_x)
        # print type(rf.predict_proba(n_x))
        predict_record_all = g.machinelearning[num][-1].predict_proba(n_x)
        print
        predict_record_all
        predict_record = []
        for i in range(len(predict_record_all)):
            predict_record.append(predict_record_all[i][2])  # 多分类时，这里需要修改
        # np.array(predict_record)
        predict_record = np.array(predict_record)
        stocks_rank = np.argsort(predict_record)
        """
        for i in range(len(stocks_rank)):
            if stocks_rank[i] > len(predict_record)-50: #>len(predict_record)-50  or < 50
                stocks_l.append(list(x.index)[i])
        print '买入股票数：',len(stocks_l)
        return stocks_l
        """
        df_AllFactor = DataFrame()
        predict_record = np.array(predict_record)
        predict_record = pd.Series(predict_record)
        df_AllFactor['AllFactor'] = predict_record
        stocks_rank = np.array(stocks_rank)
        stocks_rank = pd.Series(stocks_rank)
        df_AllFactor['AllFactor_sorted_rank'] = stocks_rank
        df_AllFactor.index = x.index
        return df_AllFactor
    else:
        return False


# 东方证券机器学习因子库因子，集成到一起
def get_df_AllFactor(stock_list, context, asc, num):
    # 计次
    # g.contrl_choose_name = ['linear_regression_choose','optimal_choose']
    # 控制选股方法
    # g.contrl_choose = [0,0,0]
    # 计算因子,并将因子值存入全局变量
    if g.cal_factor:
        factor_cal(stock_list, context, asc, num)
        g.list_count_set[num] = g.list_count_set[num] + 1
        print
        '各周期因子数量:' + str(len(g.list_panel_set[0])) + ' ' + str(len(g.list_panel_set[1])) + ' ' + str(
            len(g.list_panel_set[2]))
    # 线性回归选股
    df_AllFactor = globals()[g.contrl_choose_name[g.contrl_choose[num]]](stock_list, context, asc, num)
    """
    if g.linear_regression_choose:
        df_AllFactor=linear_regression_choose(stock_list, context, asc,num)
    # 最优化选股
    if g.optimal_choose:
        df_AllFactor=optimize_choose(stock_list, context, asc,num)
    """
    # if type(df_AllFactor) != bool:
    #    print '周期',num,len(df_AllFactor)
    return df_AllFactor


'''
================================================================================
每天收盘后
================================================================================
'''


# 每日收盘后要做的事情（本策略中不需要）
def after_trading_end(context):
    return
