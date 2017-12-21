#coding:utf-8
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
import sys
import gc
import re

filedir ='/hdd/ctp/day/'
outputdir = u'/media/charles/common_file/python/quant/intern/hand out'



start_date, end_date, test_date ,period, target, method = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
#start_date, end_date, period, target = '20171101', '20171109', '1s', 'hc'
days = pd.date_range(start=start_date, end=end_date, freq = 'B')
dayLst = []
for day in days:
    temp = day.strftime('%Y-%m-%d').split('-')
    day = temp[0]+temp[1]+temp[2]
    dayLst.append(day)
testLst = []
testLst.append(test_date)


def findType(target = target):
    noble_metal = ['au', 'ag']; noble_metal.extend(map(str.upper, noble_metal))
    nonferrous_metal = ['sf', 'sm', 'cu', 'ni', 'pb', 'sn', 'zn', 'al']; nonferrous_metal.extend(map(str.upper, nonferrous_metal))
    black_metal = ['hc', 'rb', 'zc', 'i1', 'j1', 'jm']; black_metal.extend(map(str.upper, black_metal))
    farm_produce = ['cf', 'oi', 'rm', 'a1', 'c1', 'cs', 'jd', 'm1', 'p1', 'y1', 'wh', 'b1', 'rs', 'lr', 'ri', 'cy',
                    'wr', 'fb', 'pm', 'jr']; farm_produce.extend(map(str.upper, farm_produce))
    chemcical_produces = ['fg', 'ma', 'bu', 'l1', 'pp', 'ru', 'v1', 'fu', 'bb', 'ta']; chemcical_produces.extend(map(str.upper, chemcical_produces))
    index_futures = ['ic', 'if', 'ih']; index_futures.extend(map(str.upper, index_futures))
    govern_loan = ['t1', 'tf']; govern_loan.extend(map(str.upper, govern_loan))
    typeLst = [noble_metal, nonferrous_metal, black_metal, farm_produce, chemcical_produces, index_futures, govern_loan]
    for typename in typeLst:
        if target[:2] in typename:
            return typename


def singleType(df, lst, period, typename = 'noble' ): #计算同一个类别下各主期权的相关系数，如有色金属，贵金属
    noble_metal = ['au','ag']
    nonferrous_metal = ['sf', 'sm', 'cu', 'ni', 'pb', 'sn', 'zn', 'al']
    black_metal = ['hc','rb','zc', 'i1', 'j1', 'jm']
    farm_produce = ['cf', 'oi', 'rm', 'a1', 'c1', 'cs', 'jd', 'm1', 'p1', 'y1', 'wh', 'b1', 'rs', 'lr', 'ri', 'cy', 'wr', 'fb','pm','jr']
    chemcical_produces = ['fg', 'ma', 'bu', 'l1', 'pp', 'ru', 'v1', 'fu', 'bb', 'ta']
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


def processData(filedir=filedir, dayLst=dayLst, period=period, keywd='mid_price_return', outputdir=outputdir,
                anatype='type'):
    day = dayLst[0]
    date = dayLst[0]
    dir = filedir + day + '.dat.gz'

    temp = pd.read_csv(dir, header=None, index_col=0, compression='gzip',
                       names=['ticker', 'bid_price', 'bid_volume', 'ask_price', 'ask_volume', 'last_price',
                              'last_volume', 'open_interest', 'turnover'])
    temp = temp.iloc[1:(temp.shape[0] - 5), :]  # drop first record and last record of the day
    timeIndex(temp, day)
    temp.sort_index(inplace=True)
    temp = temp.iloc[1:-1, :]
    major = findMostInType(temp)
    corrmat = corrMat(temp, lst=major.values(), freq=period, keywd=keywd)

    if len(days) > 1:
        for day in dayLst[1:]:
            dir = filedir + day + '.dat.gz'
            temp = pd.read_csv(dir, header=None, index_col=0,
                               names=['ticker', 'bid_price', 'bid_volume', 'ask_price', 'ask_volume', 'last_price',
                                      'last_volume', 'open_interest', 'turnover'])
            temp = temp.iloc[1:(temp.shape[0] - 5), :]  # drop first record and last record of the day
            timeIndex(temp, day)
            temp.sort_index(inplace=True)
            major = findMostInType(temp)
            corrmat0 = corrMat(df=temp, lst=major.values(), freq=period, keywd=keywd)
            corrmat = pd.concat([corrmat, corrmat0])
            del corrmat0;
            gc.collect()
        date = dayLst[0] + ' to ' + dayLst[-1]
    if anatype == 'type':
        typeLst = findType(target)  # get the type of target
        anaLst = []
        for option in major.values():
            if option[:2] in typeLst:
                anaLst.append(option)
        corrmat = corrmat.loc[:,anaLst]
    else:
        tarCol = target_col(corrmat= corrmat)
        corrmat = findNstElem(corrmat, ticker=tarCol)
    if 'ind' in corrmat.columns.values:
        corrmat.drop('ind', axis=1, inplace=True)
    return corrmat


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


def findNstElem(retmat, ticker=target ,k = 10):   #找出与单一期权相关度最高的k个
    #correlation to specified ticker
    cols = retmat.corr().nlargest(k, ticker)[ticker].index
    return retmat[cols]


def target_col(corrmat ,target = target):   #获得target对应的主力合约名字
    for col in corrmat.columns:
        if target in col or target.upper() in col[:2]:
            return col

def train_and_test(trainLst, testLst, method):
    train = processData(filedir= filedir, dayLst = trainLst, period=period, anatype='type',keywd='mid_price_return')
    train.dropna(axis=0, how='all', inplace=True)
    train.dropna(axis=1, how='all', inplace=True)
    #train = (train - train.mean())/ train.std()
    train.fillna(method= 'ffill', inplace=True)
    train.fillna(method= 'bfill', inplace=True)
    targetCol = target_col(train)
    shift_period = 0 - int(period[:-1])
    train_y = pd.DataFrame(train[targetCol]).shift(shift_period).fillna(method='ffill').values
    class_train_y = [1 if i > 0 else 0 for i in train_y]
    if targetCol in train.columns.values:
        train_drop = train.drop(targetCol, axis=1)
    train_X = train_drop.values
    if method == 'LR':
        clf = LR()
        clf.fit(train_X, class_train_y)
        print clf.score(train_X, class_train_y)


    test = processData(filedir=filedir, dayLst=testLst, period=period, anatype='type', keywd='mid_price_return')
    test.dropna(axis=0, how='all', inplace=True)
    test.dropna(axis=1, how='all', inplace=True)
    #test = (test - test.mean()) / test.std()
    test.fillna(method='ffill', inplace=True)
    test.fillna(method='bfill', inplace=True)
    targetCol = target_col(test)
    shift_period = 0 - int(period[:-1])
    test_y = pd.DataFrame(test[targetCol]).shift(shift_period).fillna(method='ffill').values
    class_test_y = [1 if i > 0 else 0 for i in test_y]
    if targetCol in test.columns.values:
        test_drop = test.drop(targetCol, axis=1)
    test_X = test_drop.values

    test_predict = clf.predict(test_X)
    print clf.score(test_X, class_test_y)


train_and_test(trainLst=dayLst, testLst=testLst, method= method)
