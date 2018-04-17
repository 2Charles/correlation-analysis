#coding:utf-8
import pre_process
import pandas as pd
import os
import MySQLdb
import numpy as np
import gc


class automation(object):

    def __init__(self, dir='/hdd/ctp/day/', save=True, split=2):
        self.dir = dir
        self.save = save
        self.out_dir = '/media/sf_ubuntu_share/corr_output/'
        self.split = split
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        self.calculated = []
        self.uncalculated = []
        self.multi_uncalculated = {}
        self.multi_calculated = {}
        self.conn = MySQLdb.connect(host='47.52.224.156', user='hhui', passwd='hhui123456')
        self.cursor = self.conn.cursor()
        self.cursor.execute("""create database if not exists db_corr""")
        self.conn.select_db('db_corr')
        self.cursor.execute("""create table if not exists final_corr
            (start_date DATE not null,
            end_date DATE not null,
            ticker1 varchar(32) not null,
            ticker2 varchar(32) not null,
            type SMALLINT not null DEFAULT 0,
            period INT not null,
            lag INT not null,
            corr DOUBLE,
            symbol1 varchar(32),
            symbol2 varchar(32),
            primary key(start_date, end_date, ticker1, ticker2, type, lag, period))""")
        self.cursor.execute("""create table if not exists multi_days_corr
                    (start_date DATE not null,
                    end_date DATE not null,
                    duration INT not null,
                    ticker1 varchar(32) not null,
                    ticker2 varchar(32) not null,
                    type SMALLINT not null DEFAULT 0,
                    period INT not null,
                    lag INT not null,
                    corr DOUBLE,
                    primary key(start_date, end_date, ticker1, ticker2, type, lag, period))""")
        self.cursor.execute("""create table if not exists ticker_symbol
                            (start_date DATE not null,
                            end_date DATE not null,
                            ticker varchar(32) not null,
                            symbol varchar(32),
                            primary key(start_date, end_date, ticker))""")

    def get_calculated(self):  # find calculated date from database
        self.cursor.execute('select * from final_corr group by start_date')
        ans = self.cursor.fetchall()
        for row in ans:
            tmp = str(row[0]).split('-')
            self.calculated.append(tmp[0] + tmp[1] + tmp[2] + '.dat.gz')

    def get_uncalculated(self):  # contrast with local data file and calculated list to get un_calculated data
        for file in os.listdir(self.dir):
            if self.filter_by_size(file) and file not in self.calculated:
                self.uncalculated.append(file)
        if '20180416.dat.gz' not in self.uncalculated:
            self.uncalculated.append('20180416.dat.gz')

    def get_k_days_calculated(self, k=5):
        self.cursor.execute(
            'select end_date from multi_days_corr where duration = ' + str(k) + ' group by start_date')
        ans = self.cursor.fetchall()
        tmplst = []
        for row in ans:
            tmp = str(row[0]).split('-')
            tmplst.append(tmp[0] + tmp[1] + tmp[2] + '.dat.gz')
            self.multi_calculated[k] = tmplst

    def get_k_days_uncalculated(self, k=5):  # 按照最后一天是否计算过为标准，如20180416-20180420是一个组合
        if k not in self.multi_uncalculated.keys():
            self.multi_uncalculated[k] = []
        for file in os.listdir(self.dir):
            if self.filter_by_size(file) and file not in self.multi_calculated[k]:
                self.multi_uncalculated[k].append(file)

    def calculate_single_day(self):
        self.uncalculated.sort(reverse=True)
        for file in self.uncalculated:
            if self.filter_by_size(file):
                date = file.split('.')[0]
                print 'calculating day:', date
                for type in [0, 1]:
                    if type == 0:
                        periodlst = ['0s', '1s', '5s']
                        laglst = ['0s', '500ms', '1s', '5s']
                        resample_periodlst = ['1s', '5s', '10s', '30s']
                        keywd = 'rolling'
                    else:
                        periodlst = ['0s']
                        laglst = ['0s', '500ms', '1s', '5s']
                        resample_periodlst = ['1s', '5s', '10s', '30s']
                        keywd = 'aggravated'
                    corr = pre_process.pre_process(filedir=self.dir, type=type, save=self.save)
                    raw_data = corr.loaddata(date)
                    data_dic = corr.trim_merge(raw_data)
                    for period in periodlst:
                        calculated_dic = corr.calc_all_ticker(data_dic, period, resample_periodlst)
                        return_df = corr.merge_return(calculated_dic, period, keywd=keywd)
                        corr_df = return_df['no_resample']  # only concern un-resampled data to calculate corr
                        del return_df
                        gc.collect()
                        target_lst = corr_df.columns.values
                        for lag in laglst:
                            # compare lag and period, lag should less than period.
                            # if type == 0:  # only when type == 1 need to compare period and lag as when type == 1, periodlst is constrained to be ['0s'] but laglst should be all of four.
                            if not self.compare(period, lag):
                                continue
                            print pd.datetime.now(), 'lag: ', lag, ' period: ', period
                            for target in target_lst:
                                this_shifted = self.shift(corr_df, target, lag)
                                if self.save:
                                    if not os.path.exists(self.out_dir+'lagged_return/'+date+'/'+keywd+'/'+target+'/'):
                                        os.makedirs(self.out_dir+'lagged_return/'+date+'/'+keywd+'/'+target+'/')
                                    this_shifted.to_csv(self.out_dir+'lagged_return/'+date+'/'+keywd+'/'+target+'/'+'period_'+period+'_'+'lag_'+lag+'.dat.gz', compression='gzip')
                                level = 0
                                symbol1 = corr.symbolDict[level][date][target[:2]]
                                self.ticker_symbol(start_date=date, end_date=date, ticker=target, symbol=symbol1)
                                corr_mat = this_shifted.corr()
                                corr_mat.fillna(-2, inplace=True)
                                if not os.path.exists(
                                        self.out_dir + '/corr/' + date + '/' + keywd + '/' + target + '/'):
                                    os.makedirs(self.out_dir + '/corr/' + date + '/' + keywd + '/' + target + '/')
                                corr_mat.to_csv(
                                    self.out_dir + '/corr/' + date + '/' + keywd + '/' + target + '/' + period + '_' + lag + '.dat.gz', compression='gzip')
                                if lag == '500ms':
                                    lag_insert = '500s'
                                else:
                                    lag_insert = lag
                                for ticker2 in corr_mat.index.values:
                                    corr_value = corr_mat[target][ticker2]
                                    symbol2 = corr.symbolDict[level][date][ticker2[:2]]
                                    self.cursor.execute("""REPLACE INTO final_corr(
                                                            start_date,
                                                            end_date,ticker1,
                                                            symbol1,
                                                            ticker2,
                                                            symbol2,
                                                            type,
                                                            period,
                                                            lag,
                                                            corr)
                                                            VALUES ('%s','%s','%s','%s','%s','%s','%d','%d','%d','%.8f')"""
                                                        % (date, date, target, symbol1, ticker2, symbol2, type,
                                                           int(period[:-1]), int(lag_insert[:-1]), corr_value))
                                    self.conn.commit()
            self.calculated.append(file)
            print 'done'

    def calculate_k_days(self, k=5):
        self.multi_uncalculated[k].sort(reverse=True)
        for file in self.multi_uncalculated[k]:
            if self.filter_by_size(file):
                day = file.split('.')[0]
                cal_range = []
                date_rng = pd.date_range(end=day, periods=k, freq='B')
                for day in date_rng:
                    try:
                        tmp = (str(day).split(' ')[0]).split('-')
                        day = tmp[0] + tmp[1] + tmp[2]
                        if self.filter_by_size(day + '.dat.gz'):
                            cal_range.append(day)
                    except:
                        print 'soomething wrong with date: ', day
                # common_ticker = self.get_k_days_common_ticker(cal_range)
                for type in [0, 1]:
                    if type == 0:
                        keywd = 'rolling_return'
                        periodlst = ['0s', '1s', '5s']
                        laglst = ['0s', '500ms', '1s', '5s']
                    else:
                        keywd = 'aggravated_return'
                        periodlst = ['0s']
                        laglst = ['0s', '500ms', '1s', '5s']
                    df_dic = {} # 用于存储每天return df，因为可能不同天数里的ticker不一样，所以要先找交集
                    col_dic = {} # 存储每天的ticker list，也就是dataframe的columns
                    for period in periodlst:
                        for i in range(len(cal_range)):
                            day = cal_range[i]
                            data_dir = '/media/sf_ubuntu_share/corr_output/return/' + day + '/' + keywd + '/period_' + period + '/' + 'no_resample.dat.gz'  # 待补充  默认是用新的数据存储格式，即../price/date/ticker/period_0s.csv
                            tmp = pd.read_csv(data_dir, header=0, index_col=0, compression='gzip')
                            tmp.index = pd.DatetimeIndex(tmp.index.values)
                            # target_col = tmp[keywd].values
                            df_dic[day] = tmp
                            if i == 0:
                                common_set = set(tmp.columns.values)
                            else:
                                common_set = common_set & set(tmp.columns.values)
                        this_df = pd.DataFrame()
                        for day in cal_range:  # 确保日期是有序的
                            tmp = df_dic[day]
                            tmp = tmp[list(common_set)]
                            this_df = pd.concat([this_df, tmp])  #把多天的returnconcat在了一起
                        for lag in laglst:
                            if lag == '500ms':
                                to_lag = self.split / 2
                                lag_insert = '500s'
                            else:
                                to_lag = int(lag[:-1]) * self.split  # 需要lag的行数
                                lag_insert = lag
                            for target in list(common_set):
                                shifted = this_df.copy()
                                target_col = shifted[target].values
                                target_col = list(target_col[to_lag:])
                                for i in range(to_lag):
                                    target_col.append(-2)
                                shifted[target] = target_col
                                shifted.replace(-2, np.nan, inplace=True)
                                shifted.fillna(method='ffill', inplace=True)
                                corr_mat = shifted.corr()
                                del shifted
                                gc.collect()
                                for ticker2 in corr_mat.columns.values:
                                    corr_value = corr_mat[target][ticker2]
                                    self.cursor.execute(
                                        """REPLACE INTO multi_days_corr(start_date,end_date,duration,ticker1,ticker2,type,period,lag,corr)VALUES ('%s','%s','%d','%s','%s','%d','%d','%d','%.8f')""" % (
                                            cal_range[0], cal_range[-1], k, target, ticker2, type,
                                            int(period[:-1]), int(lag_insert[:-1]), corr_value))
                                    self.conn.commit()
                print 'done'
            print 'done'

    def compare(self, period, lag):
        if lag[-2:] == 'ms':
            lag_ = int(lag[:-2])
        else:
            lag_ = int(lag[:-1]) * 1000
        period_ = int(period[:-1]) * 1000
        flag = lag_ <= period_
        if period == '0s' and lag == '500ms':
            flag = True
        if lag == '500ms' and period != '0s':  # only use lag = 500ms when period = 0s
            flag = False
        return flag

    # def shift(self, data, target, lag, corr):
    #     align_base = corr.get_align_base(data)
    #     res = corr.shift_align(data, target, lag, align_base=align_base)
    #     return res
    def shift(self, data, target, lag):
        res = data.copy()
        if 'ms' in lag:
            to_lag = float(int(lag[:-2])) / 1000
        elif lag[-1] == 's':
            to_lag = float(int(lag[:-1])) / 1000
        else:  # time unit: minute(m)
            to_lag = float(int(lag[:-1])) / 1000 * 60
        to_lag = int(to_lag)
        tmp = res[target].values
        tmp = list(tmp[to_lag:])
        for i in range(to_lag):
            tmp.append(-2)
        res[target] = tmp
        res.replace(-2, np.nan, inplace=True)
        res.fillna(method='ffill', inplace=True)
        return res

    def sample(self, data, period, target, corr):
        res = corr.sampledata(data, period, target)
        res.dropna(how='all', axis=0, inplace=True)
        res.fillna(method='ffill', inplace=True)
        res.fillna(method='bfill', inplace=True)
        return res

    def filter_by_size(self, file):
        '''only calculate those files that have size bigger than 1Mb '''
        return os.path.getsize(self.dir + '/' + file) / float(1024 * 1024) > 1

    def ticker_symbol(self, start_date, end_date, ticker, symbol):
        self.cursor.execute('''REPLACE INTO ticker_symbol(
        start_date,
        end_date,
        ticker,
        symbol)
        values('%s','%s','%s','%s')''' % (start_date, end_date, ticker, symbol))
        self.conn.commit()

    # def get_k_days_common_ticker(self, daylst):  # 获取k天范围内的共同ticker，读取它们然后concat用于求k天的corr
    #     common_ticker = set()
    #     for i in range(len(daylst)):
    #         this_day = []
    #         day = daylst[i]
    #         if isinstance(day, int):
    #             day = str(day)
    #         self.cursor.execute('select ticker from ticker_symbol where start_date = ' + day)
    #         ans = self.cursor.fetchall()
    #         for row in ans:
    #             this_day.append(row[0])
    #         if i == 0:  # 第一天，并集，后面用交集
    #             common_ticker = set(this_day)
    #         else:
    #             common_ticker = common_ticker & set(this_day)
    #     return list(common_ticker)


if __name__ == '__main__':
    auto = automation()
    auto.get_calculated()
    auto.get_uncalculated()
    auto.get_k_days_calculated()
    auto.get_k_days_uncalculated()
    print auto.uncalculated
    auto.calculate_single_day()
