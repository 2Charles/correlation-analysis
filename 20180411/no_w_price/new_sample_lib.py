import re
import pandas as pd
import numpy as np

class sample(object):
    def __init__(self, period, split=2):
        self.period = period
        self.split = split

    def extract_single_day(self, data, day):
        '''get single data from a concated multiday data'''
        timerange1 = pd.date_range(day+' 09', day+' 11:30', freq=str(1000/self.split)+'ms')
        timerange2 = pd.date_range(day + ' 13:30', day + ' 15', freq=str(1000/self.split)+'ms')
        flag = map(lambda x: (x in timerange1) or (x in timerange2), data.index.values)
        if np.sum(flag) == 0:
            print 'your data contains no records of day:', day
        else:
            return data[flag]

    def sample_single_day(self, data, day):
        # devide into morning data and afternoon data and calculate spreately
        if self.period[-2:] == 'ms':
            step = int(self.period[:-2]) *1.0 / 1000 * self.split
            step = int(step)
        elif self.period[-1] == 's':
            if self.period == '0s':
                step = 1
            else:
                step = int(self.period[:-1]) * self.split
        elif self.period[-1] == 'm':
            step = int(self.period[:-1]) * 60 * self.split
        else:
            print 'error period unit, it should be ms, s or m'
        morning_rng = pd.date_range(day + ' 09', day + ' 11:30', freq=str(1000/self.split)+'ms')
        morning_flag = [True if time in morning_rng else False for time in data.index.values]
        afternoon_flag = [False if morning else True for morning in morning_flag]
        morning_data = data[morning_flag]
        afternoon_data = data[afternoon_flag]

        morning_size = morning_data.shape[0]
        afternoon_size = afternoon_data.shape[0]
        morning_ret, afternoon_ret = [], []
        morning_mid = morning_data['mid_price'].values
        afternoon_mid = afternoon_data['mid_price'].values
        for i in range(morning_size):
            if i < step:
                morning_ret.append(0)
            else:
                ret = (morning_mid[i] - morning_mid[i-step])/morning_mid[i-step]
                morning_ret.append(ret)

        morning_data.loc[:,'rolling_return'] = morning_ret

        for i in range(afternoon_size):
            if i < step:
                afternoon_ret.append(0)
            else:
                ret = (afternoon_mid[i] - afternoon_mid[i - step]) / afternoon_mid[i - step]
                afternoon_ret.append(ret)
        afternoon_data.loc[:,'rolling_return'] = afternoon_ret
        res = pd.concat([morning_data,afternoon_data])
        return res

    def sample_multidays(self, data):
        '''sample data manually, keep as many none-zero value as possible'''
        day_first = re.split(r'[-T]', str(data.index.values[0]))[0] + re.split(r'[-T]', str(data.index.values[0]))[1] + \
                    re.split(r'[-T]', str(data.index.values[0]))[2]
        day_last = re.split(r'[-T]', str(data.index.values[-1]))[0] + re.split(r'[-T]', str(data.index.values[0]))[1] + \
                   re.split(r'[-T]', str(data.index.values[0]))[2]
        if day_first == day_last:
            day = day_first
            res = self.sample_single_day(data=data, day=day)
        else:
            res = pd.DataFrame()
            days = pd.date_range(start=day_first, end=day_last, freq='B')
            daylst = []
            for day in days:
                temp = day.strftime('%Y-%m-%d').split('-')
                day = temp[0] + temp[1] + temp[2]
                daylst.append(day)
            for day in daylst:
                singleday = self.extract_single_day(data, day)
                temp = self.sample_single_day(data=singleday, day=day)
                res = pd.concat([res, temp])
        return res