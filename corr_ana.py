#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import re
import matplotlib.pylab as plt
import seaborn as sns
import os
import gc
import sys

#need to input script like 'python corr_ana.py '20170910' '20170920' '5s'


filedir ='/hdd/ctp/day/'
outputdir = u'/home/hui/文档/corr output/'

start_date, end_date, period = sys.argv[1], sys.argv[2], sys.argv[3]
days = pd.date_range(start=start_date, end=end_date, freq = 'B')
dayLst = []
for day in days:
    temp = day.strftime('%Y-%m-%d').split('-')
    day = temp[0]+temp[1]+temp[2]
    dayLst.append(day)


def singleType(df, lst, period, typename = 'noble' ): #计算同一个类别下各主期权的相关系数，如有色金属，贵金属
    noble_metal = ['au','ag']
    nonferrous_metal = ['cu', 'ni', 'pb', 'sn', 'zn', 'al']
    black_metal = ['sf', 'sm', 'hc','rb','zc', 'i1', 'j1', 'jm']
    farm_produce = ['cf', 'oi', 'rm', 'a1', 'c1', 'cs', 'jd', 'm1', 'p1', 'y1', 'wh', 'b1', 'rs', 'lr', 'ri', 'cy', 'wr', 'fb','pm','jr']
    chemcical_produces = ['fg', 'ma', 'bu', 'l1', 'pp', 'ru', 'v1', 'fu', 'bb','ta']
    index_futures = ['ic', 'if', 'ih']
    govern_loan = ['t1', 'tf']
    all = []
    typeLst = [noble_metal,nonferrous_metal,black_metal, farm_produce,chemcical_produces,index_futures,govern_loan]
    for elem in typeLst:
        all.extend(elem)
    all.extend(map(str.upper, all))

    res = []
    if typename == 'noble':
        noble_metal.extend(map(str.upper, noble_metal))
        for name in lst:
            if name[:2] in noble_metal:
                res.append(name)
    if typename == 'nonferrous':
        nonferrous_metal.extend(map(str.upper, nonferrous_metal))
        for name in lst:
            if name[:2] in nonferrous_metal:
                res.append(name)
    if typename == 'black':
        black_metal.extend(map(str.upper, black_metal))
        for name in lst:
            if name[:2] in black_metal:
                res.append(name)
    if typename == 'farm':
        farm_produce.extend(map(str.upper, farm_produce))
        for name in lst:
            if name[:2] in farm_produce:
                res.append(name)
    if typename == 'chemical':
        chemcical_produces.extend(map(str.upper, chemcical_produces))
        for name in lst:
            if name[:2] in chemcical_produces:
                res.append(name)

    if typename == 'futures':
        index_futures.extend(map(str.upper, index_futures))
        for name in lst:
            if name[:2] in index_futures:
                res.append(name)
    if typename == 'loan':
        govern_loan.extend(map(str.upper, govern_loan))
        for name in lst:
            if name[:2] in govern_loan:
                res.append(name)
    for name in lst:
        if name[:2] not in all:
            print 'Opps, I got ', name,' un-added for single type analysis.'
    return corrMat(df, lst=res, freq= period)


def processData(filedir = filedir, dayLst = dayLst, period = period ,keywd = 'mid_price_return', outputdir = outputdir):
    day = dayLst[0]
    date = dayLst[0]
    dir = filedir + day + '.dat.gz'

    temp = pd.read_csv(dir, header=None, index_col=0, compression = 'gzip',
                       names=['ticker', 'bid_price', 'bid_volume', 'ask_price', 'ask_volume', 'last_price',
                              'last_volume', 'open_interest', 'turnover'])
    temp = temp.iloc[1:(temp.shape[0] - 1), :]  # drop first record and last record of the day
    timeIndex(temp, day)
    temp.sort_index(inplace=True)
    temp = temp.iloc[1:-1, :]
    major = findMostInType(temp)
    corrmat = corrMat(temp, lst=major.values(), freq=period, keywd=keywd)
    lst = ['noble', 'nonferrous', 'black', 'farm', 'chemical', 'futures', 'loan']
    type0 = singleType(temp, lst=major.values(), period=period, typename=lst[0])
    type1 = singleType(temp, lst=major.values(), period=period, typename=lst[1])
    type2 = singleType(temp, lst=major.values(), period=period, typename=lst[2])
    type3 = singleType(temp, lst=major.values(), period=period, typename=lst[3])
    type4 = singleType(temp, lst=major.values(), period=period, typename=lst[4])
    type5 = singleType(temp, lst=major.values(), period=period, typename=lst[5])
    type6 = singleType(temp, lst=major.values(), period=period, typename=lst[6])
    if len(days) > 1:
        for day in dayLst[1:]:
            dir = filedir + day + '.dat.gz'
            temp = pd.read_csv(dir, header=None, index_col=0,
                               names=['ticker', 'bid_price', 'bid_volume', 'ask_price', 'ask_volume', 'last_price',
                                      'last_volume', 'open_interest', 'turnover'])
            temp = temp.iloc[1:(temp.shape[0] - 1), :]  # drop first record and last record of the day
            timeIndex(temp, day)
            temp.sort_index(inplace=True)
            major = findMostInType(temp)
            corrmat0 = corrMat(df=temp, lst=major.values(), freq=period, keywd=keywd)
            corrmat = pd.concat([corrmat, corrmat0])
            temp0 = singleType(temp, lst=major.values(), period=period, typename=lst[0])
            temp1 = singleType(temp, lst=major.values(), period=period, typename=lst[1])
            temp2 = singleType(temp, lst=major.values(), period=period, typename=lst[2])
            temp3 = singleType(temp, lst=major.values(), period=period, typename=lst[3])
            temp4 = singleType(temp, lst=major.values(), period=period, typename=lst[4])
            temp5 = singleType(temp, lst=major.values(), period=period, typename=lst[5])
            temp6 = singleType(temp, lst=major.values(), period=period, typename=lst[6])
            type0 = pd.concat([type0, temp0])
            type1 = pd.concat([type1, temp1])
            type2 = pd.concat([type2, temp2])
            type3 = pd.concat([type3, temp3])
            type4 = pd.concat([type4, temp4])
            type5 = pd.concat([type5, temp5])
            type6 = pd.concat([type6, temp6])
            del corrmat0; gc.collect()
            del temp0, temp1, temp2, temp3, temp4, temp5, temp6; gc.collect()
        date = dayLst[0] + ' to ' + dayLst[-1]
    saveFigCsv(corrmat, freq=period, output_dir=outputdir+ 'multidays/', date=date)
    saveFigCsv(type0, freq=period, output_dir= outputdir+ '/single type/' + lst[0] + '/', date=date, fontsize=25)
    saveFigCsv(type1, freq=period, output_dir=outputdir + '/single type/' + lst[1] + '/', date=date, fontsize=25)
    saveFigCsv(type2, freq=period, output_dir=outputdir + '/single type/' + lst[2] + '/', date=date, fontsize=25)
    saveFigCsv(type3, freq=period, output_dir=outputdir + '/single type/' + lst[3] + '/', date=date, fontsize=25)
    saveFigCsv(type4, freq=period, output_dir=outputdir + '/single type/' + lst[4] + '/', date=date, fontsize=25)
    saveFigCsv(type5, freq=period, output_dir=outputdir + '/single type/' + lst[5] + '/', date=date, fontsize=25)
    saveFigCsv(type6, freq=period, output_dir=outputdir + '/single type/' + lst[6] + '/', date=date, fontsize=25)



def saveFigCsv(corrmat, freq, output_dir, date,path_tail = None, figsize = (30,20), fontsize = 10):
    fig,ax = plt.subplots(figsize = figsize)
    sns.set(font_scale=1.25)
    sns.heatmap(corrmat.corr(),cmap = 'coolwarm', cbar=True, annot=True,square=True, fmt='.2f', annot_kws={'size': fontsize})
    plt.xticks(rotation=90, fontsize = fontsize)
    plt.yticks(rotation = 0, fontsize = fontsize)
    plt.title(u'correlation heatmap of major option')
    dir = output_dir + date + '/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    fig.savefig(dir + date + '_' + freq + '.jpg')
    corrmat.to_csv(dir +date+'_'+freq+'_return.csv')
    corrmat.corr().to_csv(dir + date + '_' + freq + '_corr.csv')


def timeIndex(df, date):
    lst = list(df.index.values)
    year, month, day = date[:4],date[4:6],date[6:]
    res = []
    for time in lst:
        s = re.split(r'[:.]', time)
        if int(s[-1]) <= 500:
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
        s = year + '-' + month + '-' + day + ' ' + s
        res.append(s)
    df.index = pd.DatetimeIndex(res)


def sample(df, freq, how='mean'):
    if how == 'first':
        df = df.resample(freq).first().dropna(axis=0, how='all')
    if how == 'mean':
        df = df.resample(freq).mean().dropna(axis=0, how='all')
    if how == 'last':
        df = df.resample(freq).last().dropna(axis=0, how='all')
    if how == 'olhc':
        df = df.resample(freq).olhc().dropna(axis=0, how='all')
    return df


def midPrice(df):  # 计算mid_pricr
    df.loc[:, 'mid_price'] = (df.loc[:, 'bid_price'] + df.loc[:, 'ask_price']) / 2


def midRet(df):
    res = [0]
    for i in range(1, df.shape[0]):
        if df.mid_price.values[i - 1] == 0:
            temp = 0
        else:
            temp = (df.mid_price.values[i] - df.mid_price.values[i - 1]) / df.mid_price.values[i - 1]
        res.append(temp)
    df.loc[:, 'mid_price_return'] = res


def midRetCum(df):
    df.loc[:, 'mid_price_return_cum'] = df.loc[:, 'mid_price_return'].values.cumsum()


def calcAll(df):
    midPrice(df)
    midRet(df)
    midRetCum(df)


def corrMat(df, lst, freq,  threshold=1000, keywd='mid_price_return'):  # 计算相关系数矩阵
    '''if lst == None then calculate all elements otherwise only calculate elements in lst;'''

    lst.sort()
    res = pd.DataFrame()
    elem = lst[0]
    temp = df[df['ticker'] == elem]
    calcAll(temp)
    temp = sample(temp, freq=freq)
    temp = temp.rename(columns={keywd: elem})
    res[elem] = temp[elem]
    if len(lst) > 1:
        for elem in lst[1:]:
            temp = df[df['ticker'] == elem]
            if temp.shape[0] < threshold:
                continue
            else:
                calcAll(temp)
                temp = sample(temp, freq=freq)
                temp = temp.rename(columns={keywd: elem})
                res, temp = res.align(temp, join='outer')
                res = pd.concat([res.iloc[:, :-1], temp[elem]], axis=1)
                res = res.dropna(axis=1, how='all')
                res['ind'] = res.index
                res = res.drop_duplicates(subset='ind')
    return res


def filterName(lst):  # 判断是否为期权
    '''judge option or not'''
    ans = []
    for name in lst:
        if not ('-P-' in name or '-C-' in name or 'SR' in name):
            ans.append(name)
    return ans


def findMostInType(df):  #寻找主力合约
    dic = df.groupby('ticker')['turnover'].max()
    lst = dic.index.values
    lst = filterName(lst)
    existed = []
    length = {}
    most = {}
    for name in lst:
        l = dic[name]
        if name[:2] in existed:
            if l > length[name[:2]]:
                most[name[:2]] = name
                length[name[:2]] = l
        else:
            existed.append(name[:2])
            length[name[:2]] = l
            most[name[:2]] = name
    return most


def singleoption(s,df):   #计算单一品种不同ticker间的相关系数
    ansLst = []
    for name in list(set(df.ticker)):
        if s[:2] in name:
            ansLst.append(name)
    res = corrMat(df,ansLst)

    return res




def singleExchange(data, lst, excName, period, dir = outputdir):   #查看同一交易所不同
    if excName == 'DL':
        title = ['a1', 'c1', 'cs', 'i1', 'j1', 'jd', 'jm', 'l1', 'm1',\
               'p1', 'pp', 'v1', 'y1']
    elif excName == 'ZZ':
        title = ['CF', 'FG', 'MA','OI','RM', 'SF', 'SM','TA', 'ZC']
    elif excName == 'SH':
        title = ['ag',' al', 'au', 'bu', 'cu', 'hc', 'ni', 'pb', 'rb', 'ru', 'sn', 'zn']
    else:
        title = ['IC','IF','IH', 'T1', 'TF']
    res = []
    for name in lst:
        if name[:2] in title:
            res.append(name)
    return corrMat(data, lst = res, freq = period)



def cal4singleExchange(data, lst, period, output_dir = outputdir):
    for name in ['DL', 'SH', 'ZZ', 'CH']:
        single = singleExchange(data, period= period, dir=output_dir, lst=lst, excName=name)




def findNstElem(retmat, ticker ,k = 10, plot = False, save = True):   #找出与单一期权相关度最高的k个
    #correlation to specified ticker
    cols = retmat.corr().nlargest(k, ticker)[ticker].index
    cm = np.corrcoef(retmat[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, cmap = 'coolwarm' ,annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    plt.xticks(rotation = 90)
    plt.yticks(rotation = 0)
    plt.show()


processData(filedir= filedir, dayLst = dayLst, period=period,keywd = 'mid_price_return_cum')

