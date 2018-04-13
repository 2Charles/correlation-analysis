#-*- coding:utf-8 -*-

import pandas as pd
import re
import matplotlib.pylab as plt
import seaborn as sns
import os
import gc
import new_sample_lib
reload(new_sample_lib)

class pre_process(object):
    '''need to input three parameters to initialize, type controls rolling or aggravated
    0 for rolling, 1 for aggravated;
    level : 0 for major option, 1 for secondary, 2 for third '''

    def __init__(self, filedir='/hdd/ctp/day/', type=0, split=2, save=True, out_dir='/media/sf_ubuntu_share/saved-when-calculating/trial/', level=0):
        self.filedir = filedir
        self.type = type
        self.symbolDict = {}
        self.split = split
        self.save = save
        self.out_dir = out_dir
        self.level = level

    def generateDayLst(self, start, end):
        if isinstance(start, int):
            start = str(start)
        if isinstance(end, int):
            end = str(end)
        days = pd.date_range(start=start, end=end, freq='B')
        dayLst = []
        for day in days:
            temp = day.strftime('%Y-%m-%d').split('-')
            day = temp[0]+temp[1]+temp[2]
            dayLst.append(day)
        return dayLst

    def loaddata(self, day):            # 读取单日的数据，并将时间规整化
        '''only load single day
        split controls split one sec into how many parts'''
        if isinstance(day, str):
            if '.gz' not in day:
                dir = self.filedir + day + '.dat.gz'
            else:
                print 'I am here'
                dir = self.filedir + day
        if isinstance(day, int):
            dir = self.filedir + str(day) + '.dat.gz'
        if '.gz' in dir:
            temp = pd.read_csv(dir, header=None, index_col=0, compression='gzip',
                               names=['ticker', 'bid_price', 'bid_volume', 'ask_price', 'ask_volume', 'last_price',
                                      'last_volume', 'open_interest', 'turnover'])
        else:
            temp = pd.read_csv(dir, header=None, index_col=0,
                               names=['ticker', 'bid_price', 'bid_volume', 'ask_price', 'ask_volume', 'last_price',
                                      'last_volume', 'open_interest', 'turnover'])
        self.timeIndex(temp, day)
        temp.sort_index(inplace=True)
        timerange1 = pd.date_range(day+' 09', day+' 11:30', freq=str(1000/self.split)+'ms')
        timerange2 = pd.date_range(day + ' 13:30', day + ' 15', freq=str(1000/self.split)+'ms')
        flag = map(lambda x: (x in timerange1) or (x in timerange2), temp.index.values)  # only keep data that belongs to time [09,11:30] and [13:30,15:00]
        temp = temp[flag]
        return temp

    # def load_multi_days(self, dayLst, period):    #读取多日数据，并存入到一个字典中，以对应日期为key
    #     '''load multi days raw_data and store in a dictionary, with date as key'''
    #     dic = {}
    #     for day in dayLst:
    #         tmp = self.loaddata(day)
    #         data_dic = self.trim_merge(tmp)
    #         calculated_dic = self.calc_all_ticker(data_dic, period)
    #         dic[day] = self.merge_return(calculated_dic)
    #     return dic

    def findMostInType(self, df):  #寻找主力合约 选取第二、第三通过每选出一次就把那一些从列表里去掉
        dic = df.groupby('ticker')['turnover'].max()
        lst = dic.index.values
        lst = self.filterName(lst)
        for time in range(self.level+1):
            existed = []
            length = {}
            major_dic = {}
            for name in lst:
                l = dic[name]
                if name[:2] in existed:
                    if l > length[name[:2]]:
                        major_dic[name[:2]] = name
                        length[name[:2]] = l
                else:
                    existed.append(name[:2])
                    length[name[:2]] = l
                    major_dic[name[:2]] = name
            for times in range(len(lst)):
                for elem in lst:
                    if elem in major_dic.values():
                        lst.remove(elem)
        return major_dic

    def trim_merge(self, raw_data, size_thres=1000): # 选出每个主力合约，然后时间规整、时间对齐
        major = self.findMostInType(raw_data)
        major_future = major.values()
        date = str(raw_data.index.values[0]).split('T')[0]
        date = (date.split('-'))[0]+(date.split('-'))[1]+(date.split('-'))[2]
        print 'date is ', date
        self.recordSymbol(date, major, level=self.level) # record ticker-symbol pair
        data_dic = {}
        align_base = self.get_align_base(raw_data)
        for ticker in major_future:
            tmp = raw_data[raw_data['ticker'] == ticker]
            if tmp.shape[0] < size_thres:
                continue
            else:
                tmp = self.align_drop(tmp, align_base)
                tmp.fillna(method='ffill', inplace=True)
                tmp.fillna(method='bfill', inplace=True)
                data_dic[ticker] = tmp
        return data_dic

    def calc_all_ticker(self, data_dic, period, save_col=['ticker', 'bid_price', 'ask_price', 'mid_price', 'rolling_return', 'aggravated_return']):
        major_future = data_dic.keys()
        calculated_dic = {}
        date = str(data_dic[major_future[0]].index.values[0]).split('T')[0]
        date = (date.split('-'))[0]+(date.split('-'))[1]+(date.split('-'))[2]
        for ticker in major_future:
            tmp = self.calcAll(data_dic[ticker], period)
            tmp = tmp[save_col]
            calculated_dic[ticker] = tmp
            if self.save:  # save price and return
                if not os.path.exists(self.out_dir+'/price/'+date+'/'+ticker[:2] + str(self.level)+'/'):
                        os.makedirs(self.out_dir+'/price/'+date+'/'+ticker[:2] + str(self.level)+'/')
                tmp.to_csv(self.out_dir+'/price/'+date+'/'+ticker[:2] + str(self.level)+'/period_'+period+'.csv')
        return calculated_dic


    # def resample(self, data, resample_period, method='first'):
    #     '''data should be a time indexed dataframe and method'''
    #     if 'ms' in resample_period:
    #         val = int(resample_period[:-2])
    #         if val * self.split <= 1000:
    #             pass  # 表明resample period比初始的时间间隔还小
    #         else:
    #             gap = int(val * self.split * 1.0 / 1000)   # 几个点中取一个
    #             if gap > 1:
    #                 if method == 'first':
    #                     bool_list = [True]
    #                     for i in range(1, gap):
    #                         bool_list.append(False)
    #                 if method == 'last':
    #                     bool_list = []
    #                     for i in range(1, gap-1):
    #                         bool_list.append(False)
    #                     bool_list.append(True)
    #                 else:   # 对于resample period以ms为单位
    #                     bool_list = [True]
    #                     for i in range(1, gap):
    #                         bool_list.append(False)
    #                 size = data.shape[0]
    #                 keep_flag = []
    #                 for i in range(int(size/gap)):   # 利用int对size/gap向下取整，可能导致len(keep_flag) <size
    #                         keep_flag.extend(bool_list)
    #                 flag_size = len(keep_flag)
    #                 if flag_size < size:
    #                     d = size - flag_size
    #                     for i in range(d):
    #                         keep_flag.append(True)
    #         return data[keep_flag]
    #     else:
    #         return data.resample(freq=resample_period, how=method)

    def merge_return(self, calculated_dic):
        major_future = calculated_dic.keys()
        res = pd.DataFrame()
        if self.type == 0:
            keywd = 'rolling_return'
        else:
            keywd = 'aggravated_return'
        for ticker in major_future:
            symbol = ticker[:2]+str(self.level)
            res[symbol] = calculated_dic[ticker][keywd].values
        index = calculated_dic[ticker].index.values
        res.index = index
        return res

    def period_resample_combine(self, data_dic,periodlst, resample_periodlst, how='first'):
        for period in periodlst:
            calculated_dic = self.calc_all_ticker(data_dic, period)
            res = self.merge_return(calculated_dic)
            if not os.path.exists(self.out_dir+)
            for resample_period in resample_periodlst:




    def timeIndex(self, df, date):
        '''trim time into 500ms or 250ms and change it into timeseries and set as index'''
        lst = list(df.index.values)
        year, month, day = date[:4], date[4:6], date[6:]
        res = []
        for time in lst:
            s = re.split(r'[:.]', time)
            if self.split == 2:
                if int(s[-1]) == 0:
                    s = s[0] + ':' + s[1] + ':' + s[2] + '.' + '0'
                elif 0 < int(s[-1]) <= 500:
                    s = s[0] + ':' + s[1] + ':' + s[2] + '.' + '500'
                elif int(s[-1]) < 1000:
                    s[-2] = str(int(s[-2]) + 1)
                    if int(s[-2]) == 60:
                        s[-3] = str(int(s[-3]) + 1)
                        s[-2] = '00'
                        if int(s[-3]) == 60:
                            s[-3] = '00'
                            s[-4] = str(int(s[-4]) + 1)
                    elif len(s[-2]) == 1:
                        s[-2] = '0' + s[-2]
                    s = s[0] + ':' + s[1] + ':' + s[2] + '.' + '000'
            elif self.split == 4:
                if int(s[-1]) == 0:
                    s = s[0] + ':' + s[1] + ':' + s[2] + '.' + '0'
                elif int(s[-1]) <= 250:
                    s = s[0] + ':' + s[1] + ':' + s[2] + '.' + '250'
                elif int(s[-1]) <= 500:
                    s = s[0] + ':' + s[1] + ':' + s[2] + '.' + '500'
                elif int(s[-1]) <= 750:
                    s = s[0] + ':' + s[1] + ':' + s[2] + '.' + '750'
                elif int(s[-1]) < 1000:
                    s[-2] = str(int(s[-2]) + 1)
                    if int(s[-2]) == 60:
                        s[-3] = str(int(s[-3]) + 1)
                        s[-2] = '00'
                        if int(s[-3]) == 60:
                            s[-3] = '00'
                            s[-4] = str(int(s[-4]) + 1)
                    elif len(s[-2]) == 1:
                        s[-2] = '0' + s[-2]
                    s = s[0] + ':' + s[1] + ':' + s[2] + '.' + '000'
            s = year + '-' + month + '-' + day + ' ' + s
            res.append(s)
        df.index = pd.DatetimeIndex(res)

    def recordSymbol(self, date, symbolLst, level = 0): # a dictionary record ticker and symbol, first key is level and then date
        '''record symbol and ticker'''
        if level not in self.symbolDict.keys():
            self.symbolDict[level] = {}
            self.symbolDict[level][date] = symbolLst
        else:
            self.symbolDict[level][date] = symbolLst

    def shift_align(self, data, target, lag, align_base):
        '''first shift data of target colume at lag and then align it to origin dataframe'''
        df = data.copy()
        if 'ms' in lag:
            temp = pd.DataFrame(df[target].shift(periods=-int(lag[:-2]), freq=lag[-2:]))
            temp = self.align_drop(data=temp, base=align_base)
        else:
            temp = pd.DataFrame(df[target].shift(periods=-int(lag[:-1]), freq=lag[-1]))
            temp = self.align_drop(data=temp, base=align_base)
        df[target] = temp
        df.fillna(method = 'ffill', inplace=True)
        df.fillna(method = 'bfill', inplace=True)
        return df

    def get_align_base(self, df):
        '''get index as the align base for later align'''
        align_base = pd.DataFrame([1 for i in range(df.shape[0])],index=df.index)
        align_base['helper'] = align_base.index
        align_base.drop_duplicates(subset='helper', inplace=True)
        align_base.drop('helper', axis=1, inplace=True)
        return align_base

    def align_drop(self, data, base):
        '''align target data to base index and drop duplicates'''
        df = data.copy()
        _, df = base.align(df, join='left', axis = 0)
        df = pd.DataFrame(df)
        df['helper'] = df.index
        df.drop_duplicates(subset = 'helper', inplace=True)
        df.drop('helper', axis=1, inplace=True)
        return df

    def getsymbol(self, df, ticker):    #依据symbol前两个得到对应的ticker
        '''column name according to ticker as column name maybe ru0 or ru1 or ru2 and use this function to find symbol'''
        if len(ticker) == 3:
            ticker = ticker[:2]
        if len(ticker) == 1:
            ticker = ticker + '1'
        for name in df.columns.values:
            if ticker == name[:2]:
                return name

    def midPrice(self, df):  # 计算mid_pricr,存在部分记录中bid_price或者ask_price出错的情形
        flag = (df.ask_price * df.bid_price) != 0
        if flag.all():
            df.loc[:, 'mid_price'] = (df.ask_price + df.bid_price) / 2
        else:
            bid_index, ask_index = 1, 3
            mid_price = []
            for i in range(df.shape[0]):
                if (df.iloc[i, bid_index] != 0) and (df.iloc[i, ask_index] != 0):
                    mid_price.append((df.iloc[i, bid_index] + df.iloc[i, ask_index])/2)
                elif df.iloc[i, bid_index] == 0:
                    mid_price.append(df.iloc[i, ask_index])
                elif df.iloc[i, bid_index] == 0:
                    mid_price.append(df.iloc[i, bid_index])
                else:
                    mid_price.append(0)
            df.loc[:, 'mid_price'] = mid_price
            df.mid_price.replace(0, method='ffill', inplace=True)

    def rollingRet(self, df, period):
        sample = new_sample_lib.sample(period=period, split=self.split)
        res = sample.sample_multidays(df)
        return res

    def aggravatedRet(self, df):
        data = df.copy()
        data['aggravated_return'] = ((data['mid_price'] - data['mid_price'].values[0]) / data['mid_price'].values[0]).values
        return data

    def calcAll(self, df, period):
        self.midPrice(df)
        df = self.rollingRet(df, period)
        df = self.aggravatedRet(df)
        return df

    def filterName(self, lst):  # 判断是否为期权
        '''judge whether is option or not'''
        ans = []
        for name in lst:
            if not ('-P-' in name or '-C-' in name or 'SR' in name):
                ans.append(name)
        return ans
