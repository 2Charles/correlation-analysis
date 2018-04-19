# coding:utf-8
#! /home/hui/anaconda2/bin/python
# 目标： 输入一个target,起止日期，完成以下内容:
# 1.自动选出相关度高的其余tickers，可以用数量或者corr值作为threshold(从数据库中选出)
# 2.load 对应的return数据，并给出所选tickers与target的实际相关系数，down到本地
# 3.to be added

import MySQLdb
import pandas as pd
import os
import seaborn as sns
import matplotlib.pylab as plt

class Pre_Research(object):
    def __init__(self, target, dir='/media/sf_ubuntu_share/corr_output/', split=2, num=10, type=0, host='47.52.224.156',
                 user='hhui', passwd='hhui123456', db_name='db_corr', out_dir='/media/sf_ubuntu_share/corr_output/research_preparation/'):
        self.dir = dir
        self.out_dir = out_dir
        self.split = split
        self.type = type
        self.conn = MySQLdb.connect(host=host, user=user, passwd=passwd)
        self.cursor = self.conn.cursor()
        self.conn.select_db(db_name)
        self.target = target
        self.num = num  # num is the number of train symbols that you want to get from corr database
        self.calculated = []  # 找出已经用research.py计算过的日期
        self.uncalculated = []
        self.tickerlst = []  # 由于实例化class的时候需要传入target，所以获得每天的所有主力合约

    def get_symbols(self, table_name='final_corr', period=0, lag=500, method='num', val='default'):
        '''method controls select symbols by number or value threshold, method should be num or threshold'''
        if val == 'default':   # 对于method采用'threshold'来说没有通用的threshold作为val的默认值
            method = 'num'
            val = 10
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
        if method == 'num':
            for record in ans[:val]:
                lst.append(record[1])
        else:
            for record in ans:
                if record[6] >= val:
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

    def load_return_from_local(self, date, tickerlst, period, lag, resample_period='0s',sample_how='first'): # 读取对应日期、period等之下的return，并给出heatmap图、corr_mat保存到本地
        if self.target not in tickerlst:   # 确保读到的数据里有目标ticker
            tickerlst.append(self.target)
        if self.type == 0:
            keywd = 'rolling'
        else:
            keywd = 'aggravated'
        if resample_period == '0s':
            resample_keywd = 'no_resample'
        else:
            resample_keywd = 'resample_'+resample_period
        if lag == '0s':  # 是一个完整的return file,包含了各个ticker，所以是先全部读进来然后再取出tickerlst部分
            file_dir = self.dir + '/return/'+date+'/'+keywd+'/period_'+period+'/'+resample_keywd+'.dat.gz'
            tmp = pd.read_csv(file_dir,header=0,index_col=0, compression='gzip')
            tmp = tmp[list(set(tickerlst)&set(tmp.columns.values))]  # 取tickerlst和data.columns.values的交集
            tmp.index = pd.DatetimeIndex(tmp.index.values)
            corr = tmp.corr()
            self.save_corrmat_heatmap(self.out_dir+ keywd+'/single_day_return/'+date+'/'+self.target+'/',corr,period,lag,resample_period,date)
            return tmp
        else:  # 由于lag,取self.target文件夹
            file_dir = self.dir+'/lagged_return/'+ date+'/'+keywd+'/'+self.target+'/'+'/period_'+period+'_lag_'+lag+'_resample_'+resample_period+'.dat.gz' # 由于在计算corr的时候只取了未resample的，因此lagged return中全部未resample,所以在这里读取数据后补充resample
            tmp = pd.read_csv(file_dir, header=0, index_col=0, compression='gzip')
            tmp = tmp[list(set(tickerlst)&set(tmp.columns.values))]
            tmp.index = pd.DatetimeIndex(tmp.index.values)
            if resample_period == '0s':
                sampled = tmp
            else:
                sampled = tmp.resample(resample_period, sample_how)
            timerange1 = pd.date_range(date + ' 09', date + ' 11:30', freq=str(1000 / self.split) + 'ms')
            timerange2 = pd.date_range(date + ' 13:30', date + ' 15', freq=str(1000 / self.split) + 'ms')
            flag = map(lambda x: (x in timerange1) or (x in timerange2),
                       sampled.index.values)  # only keep data that belongs to time [09,11:30] and [13:30,15:00]
            sampled = sampled[flag]
            corr = sampled.corr()
            self.save_corrmat_heatmap(self.out_dir + keywd + '/single_day_return/' +date+'/'+self.target+'/', corr,
                                      period, lag, resample_period, date)
            return sampled

    def load_train_data(self, daylst, periodlst=['0s', '1s', '5s'], laglst=['0s', '500ms', '1s', '5s'], resample_list = ['0s','1s', '5s', '10s', '30s']):
        if self.check_calculated(daylst):
            print 'calculated'
        else:
            if self.type==1:  # aggravated 只计算period为0
                periodlst=['0s']
                keywd = 'aggravated'
            else:
                keywd = 'rolling'
            tickerlst = self.get_symbols()
            for period in periodlst:
                for lag in laglst:
                    for resample_period in resample_list:
                        if 'ms' in lag and period != '0s':
                            continue
                        elif 'ms' not in lag and int(lag[:-1]) > int(period[:-1]):
                            continue
                        this_df = pd.DataFrame()
                        for day in daylst:
                            tmp = self.load_return_from_local(day, tickerlst, period, lag, resample_period)
                            this_df = pd.concat([this_df, tmp])
                        if len(daylst) > 1: # 单日数据的在load return from local中已经保存
                            corr = this_df.corr()
                            self.save_corrmat_heatmap(
                                self.out_dir + keywd + '/single_day_return/'  + daylst[0] + '_' + daylst[
                                        -1]+'/'++ self.target, corr, period, lag, resample_period, daylst)

    def write_description(self, dir):
        fname = dir+'description.txt'
        if not os.path.exists(fname):
            f = open(fname,'w')
            f.write('files are named as period_lag_resample-period.dat.gz, 0s_500ms_1s means period at 0s and lag at 500ms and resample at 1s.')
            f.close()

    def save_corrmat_heatmap(self, dir, corr, period, lag, resample_period, day_or_daylst):
        if not os.path.exists(dir+'/corr_mat/'):
            os.makedirs(dir+'/corr_mat/')
        corr.to_csv(dir+'/corr_mat/' + period + '_' + lag + '_' + resample_period + '.dat.gz', compression='gzip')
        self.write_description(dir+'/corr_mat/')

        if not os.path.exists(dir + '/heatmap/'):
            os.makedirs(dir + '/heatmap/')
        plt.figure(figsize=(15, 12))  # this block of code is to save heatmap and corr_mat
        sns.heatmap(corr, cmap='coolwarm', annot=True, square=True)
        if isinstance(day_or_daylst, str):
            title = 'corr heatmap of date: ' + day_or_daylst + ' for ticker ' + self.target
        else:
            title = 'corr heatmap of date: ' + day_or_daylst[0] + '-' + day_or_daylst[-1] + ' for ticker ' + self.target
        plt.title(title)
        plt.savefig(dir + '/heatmap/' + period + '_' + lag + '_' + resample_period + '.jpg')
        plt.close()
        self.write_description(dir + '/heatmap/')

    def check_calculated(self, daylst):
        if len(daylst) == 1:
            folder_name = 'single_day_return/'+daylst[0]+'/'+self.target+'/'
        else:
            folder_name = 'multi_day_return/'+daylst[0]+'-'+daylst[-1]+'/'+self.target+'/'
        if self.type == 0:
            keywd = 'rolling'
        else:
            keywd = 'aggravated'
        if os.path.exists(self.out_dir+keywd+'/'+folder_name+'/'):
            return True
        else:
            return False

    def get_daylst_calculated(self):
        try:
            for calculated in os.listdir(self.out_dir+'/rolling/single_day_return/'):
                self.calculated.append(calculated)
        except:
            self.calculated=[]

    def get_daylst_uncalculated(self):
        for day in os.listdir(self.dir + '/corr/'):
            if day not in self.calculated:
                self.uncalculated.append(day)

    def get_tickerlst(self, date):
        self.tickerlst = []
        for ticker in os.listdir(self.dir+'/corr/'+date+'/rolling/'):
            self.tickerlst.append(ticker)


if __name__ == '__main__':
    rese = Pre_Research(target='ru0')
    rese.get_daylst_calculated()
    rese.get_daylst_uncalculated()
    for date in rese.uncalculated:
        print '----------------calculating day : %s-------------------' % date
        rese.get_tickerlst(date)
        for type in [0, 1]:
            for ticker in rese.tickerlst:
                new_rese = Pre_Research(target=ticker, type=type)
                new_rese.load_train_data([date])
