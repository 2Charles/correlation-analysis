#coding:utf-8
#! /home/hui/anaconda2/bin/python

import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import pre_research
import os

ticker1 = 'ru0'

class Research(object):

    def __init__(self, target=ticker1, saved_dir='/media/sf_ubuntu_share/corr_output/', raw_dir='/hdd/ctp/day/day/', type=0):
        self.target = target
        self.saved_dir = saved_dir
        self.raw_dir = raw_dir
        self.type = type

    def select_train_tickers(self, daylst, num=10, period='0s', lag='500ms', resample_period='0s'):
        if isinstance(daylst, str):
            daylst = [daylst]
        if resample_period == '0s':
            resample_keywd = 'no_resample'
        else:
            resample_keywd = 'resample_'+resample_period
        if len(daylst) == 1:
            folder_keywd = 'single_day_return'
            date = daylst[0]
        else:
            folder_keywd = 'multi_days_return'
            date = daylst[0]+'-'+daylst[-1]
        if self.type == 0:
            keywd = 'rolling'
        else:
            keywd = 'aggravated'
        try: # 从已经保存好的里面读取
            file_dir = self.saved_dir+'/research_preparation/'+keywd+'/'+folder_keywd+'/'+date+'/'+self.target+'/corr_mat/'
            corr = pd.read_csv(file_dir+period+'_'+lag+'_'+resample_period+'.dat.gz', header=0, index_col=0, compression='gzip')
            if corr.shape[0] < num: # 可能保存的数据中用到的ticker数少于指定的， 考虑在保存的时候保存所有ticker的
                pre = pre_research.Pre_Research(self.target)
                tickerlst = pre.get_symbols(method='num', val=num)
                df = pd.DataFrame()
                for day in daylst:
                    file_dir = self.saved_dir + '/lagged_return/' + day + '/' + keywd + '/' + self.target + '/'
                    tmp = pd.read_csv(file_dir + 'period_' + period + '_lag_' + lag + '_' + resample_keywd + '.dat.gz',
                                      header=0, index_col=0, compression='gzip')
                    tmp = tmp[tickerlst]
                    df = pd.concat([df, tmp])  # 读入了对应的日期范围内的return数据
                corr = df.corr()
                print corr[self.target]
                if not os.path.exists(self.saved_dir+'/tickers_select/'+date+'/'+self.target+'/'):
                    os.makedirs(self.saved_dir+'/tickers_select/'+date+'/'+self.target+'/')
                corr.to_csv()
            else:
                print corr[self.target]
        except:  # 没有对应的文件，需要读取后计算
            pre = pre_research.Pre_Research(self.target)
            tickerlst = pre.get_symbols(method='num', val=num)
            df = pd.DataFrame()
            for day in daylst:
                file_dir = self.saved_dir + '/lagged_return/' + day+'/'+keywd+'/'+self.target+'/'
                tmp = pd.read_csv(file_dir+'period_'+period+'_lag_'+lag+'_'+resample_keywd+'.dat.gz', header=0, index_col=0, compression='gzip')
                tmp = tmp[tickerlst]
                df = pd.concat([df, tmp])   # 读入了对应的日期范围内的return数据
            corr = df.corr()
            print corr[self.target]
