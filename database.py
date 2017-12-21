#-*- coding:utf-8 -*-
#need to input script like 'python corr_ana.py '20170910' '20170920' '5s'

# set parameters
filedir ='/hdd/ctp/day/'
start_date = '20171101'
end_date = '20171201'
type = 0     # 1 for aggravated, 0 for rolling
ticker1 = 'ni0'
outputdir = u'/home/hui/文档/corr output/'
lagLst = ['1s','5s','10s','30s','60s']
periodLst = ['1s','5s','10s','30s','60s']

# typelst = ['noble', 'nonferrous', 'black', 'farm', 'chemical', 'futures', 'loan']

import pandas as pd
import numpy as np
import re
import matplotlib.pylab as plt
import seaborn as sns
import os
import gc
import corrlab
import MySQLdb


def createTable():
    conn = MySQLdb.connect(host = 'localhost',user='root',passwd='hhui123456')
    cursor = conn.cursor()
    cursor.execute("""create database if not exists db_corr""")
    conn.select_db('db_corr')
    cursor.execute("""create table if not exists tb_corr(
    start_date DATE not null,
    end_date DATE not null,
    ticker1 varchar(32) not null,
    ticker2 varchar(32) not null,
    type SMALLINT not null DEFAULT 0, 
    period INT not null,
    lag INT not null,
    corr DOUBLE,
    symbol1 varchar(32),
    symbol2 varchar(32),
    primary key(start_date, end_date, ticker1, ticker2, type, lag, period)
    )""")

def calc_target(start_date, end_date, type, ticker1,lagLst = lagLst, periodLst = periodLst, filedir = filedir, database = 'db_corr'):
    corr = corrlab.corrAna(filedir=filedir, start_date=start_date, end_date=end_date, type=type)
    conn = MySQLdb.connect(host='localhost', user='root', passwd='hhui123456')
    cursor = conn.cursor()
    conn.select_db(database)
    dayLst = corr.generateDayLst()
    for day in dayLst:   # 时间跨度为1天
        data = corr.concatdata([day])
        symbol1 = corr.symbolDict[day][ticker1[:2]]
        for lag in lagLst:
            for period in periodLst:
                res = pd.DataFrame()
                temp = data.copy()
                shifted = temp[ticker1].shift(-int(lag[:-1]), 's')
                align_base = corr.get_align_base(data)
                _, align_shifted = align_base.align(shifted, join='left', axis=0)
                temp[ticker1] = align_shifted.values
                temp = corr.sampledata(temp, period = period)
                temp.fillna(method='ffill',inplace=True)
                temp.fillna(method = 'bfill',inplace=True)
                temp_corr = temp.corr().sort_index()
                res = pd.concat([res,temp_corr[ticker1]])
                res.rename(columns = {0:day},inplace=True)
                res.fillna(-2,inplace=True)
                print 'lag is : %s, period is %s' %(lag, period)
                for ticker2 in temp_corr.index.values:
                    corr_value = res[day][ticker2]
                    ticker2 = ticker2.split('_')[0]
                    symbol2 = corr.symbolDict[day][ticker2[:2]]
                    cursor.execute("""REPLACE INTO tb_corr(
                                start_date,
                                end_date,
                                ticker1,
                                symbol1,
                                ticker2,
                                symbol2,
                                type,
                                period,
                                lag,
                                corr)
                                VALUES (
                                '%s', '%s','%s','%s','%s','%s','%d','%d','%d','%.6f'
                                )
                                """ % (day, day, ticker1, symbol1, ticker2, symbol2, type,int(period[:-1]), int(lag[:-1]), corr_value))
                    conn.commit()


def calc_target_week(start_date, end_date, type, ticker1, lagLst=lagLst, periodLst=periodLst, filedir=filedir,
                     database='db_corr'):
    '''day duration is 5 days'''
    corr = corrlab.corrAna(filedir=filedir, start_date=start_date, end_date=end_date, type=type)
    conn = MySQLdb.connect(host='localhost', user='root', passwd='hhui123456')
    cursor = conn.cursor()
    conn.select_db(database)
    dayLst = corr.generateDayLst()
    length = len(dayLst)
    for i in range(5,length):
        lst = dayLst[i-5: i]
        print 'processing ', lst[0] + '-'+lst[-1]
        data = corr.concatdata(lst)
        symbol1 = corr.symbolDict[lst[0] + '-'+lst[-1]][ticker1[:2]]
        for lag in ['1s','5s','10s','30s','60s']:
            for period in ['1s','5s','10s','30s','60s']:
                res = pd.DataFrame()
                temp = data.copy()
                shifted = temp[ticker1].shift(-int(lag[:-1]), 's')
                align_base = corr.get_align_base(data)
                _, align_shifted = align_base.align(shifted, join='left', axis=0)
                temp[ticker1] = align_shifted.values
                temp = corr.sampledata(temp, period=type)
                temp.fillna(method='ffill', inplace=True)
                temp.fillna(method='bfill', inplace=True)
                temp_corr = temp.corr().sort_index()
                res = pd.concat([res, temp_corr[ticker1]])
                res.rename(columns={0: lst[0] + '-' + lst[-1]}, inplace=True)
                res.fillna(-2, inplace=True)
                for symbol in res.index.values:
                    corr_value = res[(lst[0] + '-' + lst[-1])][symbol]
                    ticker2 = ticker2.split('_')[0]
                    symbol2 = corr.symbolDict[lst[0] + '-' + lst[-1]][ticker2[:2]]
                    cursor.execute("""INSERT INTO tb_corr(
                                                    start_date,
                                                    end_date,
                                                    ticker1,
                                                    symbol1,
                                                    ticker2,
                                                    symbol2,
                                                    type,
                                                    period,
                                                    lag,
                                                    corr)
                                                    VALUES (
                                                    '%s', '%s','%s','%s','%s','%s','%d','%d','%d','%.6f'
                                                    )
                                                    """ % (
                    lst[0], lst[-1], ticker1, symbol1, ticker2, symbol2, type, int(period[:-1]), int(lag[:-1]), corr_value))
                    conn.commit()
        # del data, target, res; gc.collect()


createTable()
calc_target(start_date, end_date, type, ticker1)
calc_target_week(start_date, end_date, type, ticker1)



