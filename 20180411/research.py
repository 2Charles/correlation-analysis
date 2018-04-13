#-coding:utf-8-#

import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
import numpy as np
import MySQLdb
import os

target = 'ru0'
symbol_lst = ['rb0', 'hc0', 'ni0', 'cu0', 'zn0', 'i10', 'j10', 'jm0', 'MA0']


class Research(object):
    def __init__(self, target, num=10, type=0, host='47.52.224.156', user='hhui', passwd='hhui123456', db_name='db_corr'):
        self.type = type
        self.conn = MySQLdb.connect(host=host, user=user, passwd=passwd)
        self.cursor = self.conn.cursor()
        self.conn.select_db(db_name)
        self.target = target
        self.num = num   # num is the number of train symbols that you want to get from corr database

    def get_symbols(self, table_name='final_corr', period=0, lag=500, num='default'):
        if num == 'default':
            num = self.num
        if isinstance(period, str):
            period = int(period)
        if isinstance(lag, str):
            lag = int(lag)
        sql = 'select ticker1, ticker2, type,period, lag, abs(AVG(corr))*100 as abs_cor, AVG(corr)* 100 as co, ' \
              'STD(corr)*100 as std from ' + table_name + ' where ticker1!=ticker2 and ticker1 = \"' + self.target + \
              '\" and type = 0 and period = ' + str(period) + ' and lag = ' + str(lag) + \
              ' group by ticker1,ticker2 order by abs_cor desc;'
        # print 'sql query sentence is :   ', sql
        self.cursor.execute(sql)
        ans = self.cursor.fetchall()
        lst = []
        for record in ans[:num]:
            lst.append(record[1])
        return lst

    def generate_daylst(self, start, end):
        if isinstance(start, int):
            start = str(start)
        if isinstance(end, int):
            end = str(end)
        days = pd.date_range(start=start, end=end, freq='B')
        daylst = []
        for day in days:
            temp = day.strftime('%Y-%m-%d').split('-')
            day = temp[0]+temp[1]+temp[2]
            daylst.append(day)
        return daylst

    def load_data(self, tickerlst, daylst, period, lag, file_dir='/media/sf_ubuntu_share/saved-when-calculating/trial/price/', split=2):   # 直接读取price,return数据，可选period是0s,1s,5s
        '''parameter period means the period for calculating rolling return'''
        if self.type == 0:
            keywd = 'rolling_return'
        else:
            keywd = 'aggravated_return'
        if lag == '500ms':
            lag = 1
        else:
            lag = int(lag[:-1])
        to_shift = lag * split  # 目标列需要shift的项数
        res = pd.DataFrame()
        if self.target not in tickerlst:
            tickerlst.append(self.target)
        for day in daylst:
            single_day = pd.DataFrame()
            for ticker in tickerlst:
                try:
                    dir = file_dir + day + '/' + ticker + '/'
                    if isinstance(period, int):
                        period = str(period) + 's'
                    temp = pd.read_csv(dir + 'period_' + period + '.csv')
                    single_day.loc[:, ticker] = temp[keywd].values
                except:
                    pass
            res = pd.concat([res, single_day])
        if self.target in res.columns.values:
            res[target] = res[target].shift(-to_shift)
            res.fillna(method='ffill', inplace=True)
        return res

    def load_price(self, daylst, lag, split=2, file_dir='/media/sf_ubuntu_share/saved-when-calculating/trial/price/'):
        if lag == '500ms':
            lag = 1
        else:
            lag = int(lag[:-1])
        to_shift = lag * split  # 目标列需要shift的项数
        res = pd.DataFrame()
        for day in daylst:
            single_day = pd.DataFrame()
            try:
                dir = file_dir + day + '/' + self.target + '/'
                temp = pd.read_csv(dir + 'period_0s.csv')
                single_day.loc[:, 'ask_price'] = temp['ask_price'].values
                single_day.loc[:, 'bid_price'] = temp['bid_price'].values
            except:
                pass
            res = pd.concat([res, single_day])
            res['ask_price'] = res['ask_price'].shift(-to_shift).values
            res['bid_price'] = res['bid_price'].shift(-to_shift).values
            res.fillna(method='ffill', inplace=True)
        return res

class Train(object):

    def __init__(self, target, split = 2):
        self.split = split
        self.target = target

    def sample(self, resample_period, df, keep_target_no_zero = True):
        '''keep_target_no_zero means whether try to keep as much non-zero values of target column or just sample
        by the same gap.'''
        gap = int(resample_period[:-1]) * self.split
        if keep_target_no_zero:
            row_to_keep = []
            times = df.shape[0] / gap
            for i in range(times):
                to_append = gap * i
                for row in range(gap*i,gap*i+gap):
                    if df[self.target].values[row] != 0:
                        to_append = row
                        continue
                row_to_keep.append(to_append)
            flag = [True if row in row_to_keep else False for row in range(df.shape[0])]
            return  df[flag]
        else:
            flag = [True if row % gap == 0 else False for row in range(1,df.shape[0]+1)]
            return df[flag]

    def split_x_y(self, df):
        cols = df.columns.values
        x_cols = []
        for elem in cols:
            if elem != self.target:
                x_cols.append(elem)
        X = df.loc[:,x_cols].values
        Y = df[self.target].values
        return X, Y


class Trade(Research):

    def fee_and_type_check(self, method='r'):
        '''method should be 'r' for read or 'w' for re-write'''
        '''check whether fee type and fee is in the database'''  # 和corr用同一个database
        self.cursor.execute("""create table if not exists future_fee
                (ticker varchar(32) not null,
                fee_type varchar(32) not null,
                fee DOUBLE ,
                primary key(ticker, fee_type))""")
        sql = 'select * from future_fee where ticker = \"' + self.target + '\"'
        self.cursor.execute(sql)
        ans = self.cursor.fetchall()
        if len(ans) == 0 or method == 'w':
            print 'no fee type record of this future!'
            print 'now input the fee type, 0 for percent, 1 for fixed fee:'
            fee_index = raw_input('0 for percent and 1 for fixed_value')
            while fee_index != '0' and fee_index != '1':
                print 'Invalid enter, 0 or 1 only!'
                fee_index = raw_input()
            if fee_index == '0':
                fee_type = 'per'
            else:
                fee_type = 'fixed_value'
            print 'now input the fee value:'
            try:
                fee = float(raw_input())
            except:
                print 'invalid input!'
                raise ValueError
            self.cursor.execute("""REPLACE INTO future_fee(ticker,fee_type,fee)
                                                    VALUES ('%s','%s','%f')"""
                                % (self.target, fee_type, fee))
            self.cursor.execute(sql)
            ans = self.cursor.fetchall()
            self.cursor.close()
            self.cursor = self.conn.cursor()
        return ans[0][1:]

    def simu_trade1(self, price_df, pred_return, buy_threshold, sell_threshold, lag=0,
                    hold_per_trade=5, hold=0, max_hold=500, money=5000000): #加入lag,lag应该和self.split相关,split控制一秒下有几个tick
        '''use ask_price and bid_price to simulate actual trade as it really is'''
        ask_price = price_df['ask_price'].values
        bid_price = price_df['bid_price'].values
        mid_price = (ask_price + bid_price) / 2
        pred_values = mid_price * pred_return
        asset = []
        try:
            fee_tup = self.fee_and_type_check()
            fee_type = fee_tup[0]
            fee = fee_tup[-1]
        except:
            print 'wrong to get fee type from database'
            fee_type = raw_input('enter fee type, per or fixed')
            fee = float(raw_input('enter your fee value'))
        for i in range(len(pred_values)):
            if pred_values[i] >= buy_threshold and hold < max_hold:
                to_add = min(hold_per_trade, max_hold-hold)
                hold += to_add
                if fee_type == 'per':   # 按比例
                    money -= ask_price[i+lag] * to_add * (1+fee)
                else:   # 每笔交易固定价格
                    money -= (ask_price[i + lag] + fee) * to_add
                asset.append(hold*mid_price[i]+money)
            if pred_values[i] <= sell_threshold:
                to_sub = min(hold_per_trade, hold)
                hold -= to_sub
                if fee_type == 'per':
                    money += bid_price[i+lag] * to_sub * (1-fee)
                else:
                    money += (bid_price[i+lag] - fee) * to_sub
                asset.append(hold * mid_price[i] + money)
        return asset

    def simu_trade2(self, price_df, pred_return, buy_threshold, sell_threshold, lag=0,
                    hold_per_trade=5, hold=0, max_hold=500, money=5000000):
        '''use mid_price to buy and sell'''
        ask_price = price_df['ask_price'].values
        bid_price = price_df['bid_price'].values
        mid_price = (ask_price + bid_price) / 2
        pred_values = mid_price * pred_return
        asset = []
        try:
            fee_tup = self.fee_and_type_check()
            fee_type = fee_tup[0]
            fee = fee_tup[-1]
        except:
            print 'wrong to get fee type from database'
            fee_type = raw_input('enter fee type, per or fixed')
            fee = float(raw_input('enter your fee value'))
        for i in range(len(pred_values)):
            if pred_values[i] >= buy_threshold and hold < max_hold:
                to_add = min(hold_per_trade, max_hold-hold)
                hold += to_add
                if fee_type == 'per':   # 按比例
                    money -= mid_price[i+lag] * to_add * (1+fee)
                else:   # 每笔交易固定价格
                    money -= (mid_price[i + lag] + fee) * to_add
                asset.append(hold*mid_price[i]+money)
            if pred_values[i] <= sell_threshold:
                to_sub = min(hold_per_trade, hold)
                hold -= to_sub
                if fee_type == 'per':
                    money += mid_price[i+lag] * to_sub * (1-fee)
                else:
                    money += (mid_price[i+lag] - fee) * to_sub
                asset.append(hold * mid_price[i] + money)
        return asset




