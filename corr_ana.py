#-*- coding:utf-8 -*-
#need to input script like 'python corr_ana.py '20170910' '20170920' '5s'

import pandas as pd
import numpy as np
import re
import matplotlib.pylab as plt
import seaborn as sns
import os
import corrlab
import gc


# set parameters
filedir = '/hdd/ctp/day/'
start_date = '20171208'
end_date = '20171215'
type = 0      # 1 for aggravated, 0 for rolling, 2 for both
ticker1 = 'ni0'
lagLst = ['1s','5s','10s','30s','60s']
periodLst = ['1s','5s','10s','30s','60s']

outputdir = u'/home/charles/python/intern/output/'
typelst = ['noble', 'nonferrous', 'black', 'farm', 'chemical', 'futures', 'loan']

analst = ['ru', 'zn', 'rb', 'jm', 'j1']  #appointed analyst

corr = corrlab.corrAna(filedir=filedir, start_date=start_date, end_date=end_date, type=type)

dayLst = corr.generateDayLst()


entire = pd.DataFrame()
entire_1s, entire_5s, entire_10s = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
for day in dayLst:
    lst = []
    lst.append(day)

    data = corr.concatdata(lst)
    print (day, 'calculate done')
    target = corr.getsymbol(data, ticker1)
    one_sec = corr.sampledata(data,period = '1s')
    one_sec_corr = one_sec.corr().sort_index()
    entire_1s = pd.concat([entire_1s,one_sec_corr[target]],axis = 1)
    entire_1s.rename(columns = {0:day},inplace=True)
    entire_1s.rename(columns={target: day}, inplace=True)

    five_sec = corr.sampledata(data, period='5s')
    five_sec_corr = five_sec.corr().sort_index()
    entire_5s = pd.concat([entire_5s, five_sec_corr[target]],axis = 1)
    entire_5s.rename(columns={0: day},inplace=True)
    entire_5s.rename(columns={target: day}, inplace=True)

    ten_sec = corr.sampledata(data, period='10s')
    ten_sec_corr = ten_sec.corr().sort_index()
    entire_10s = pd.concat([entire_10s, ten_sec_corr[target]],axis =1 )
    entire_10s.rename(columns={0: day},inplace=True)
    entire_10s.rename(columns={target: day}, inplace=True)

    # for period in ['1s', '5s', '10s']:
    #     data = corr.sampledata(data, period=period)
    #     data.sort_index()
    #     if type == 1:
    #         corrlab.saveFigCsv(data, period, output_dir=u'/home/charles/python/intern/output/aggravated/single day/', date=day)
    #     elif type == 0:
    #         corrlab.saveFigCsv(data, period, output_dir=u'/home/charles/python/intern/output/rolling/single day/', date=day)
    #
    #     klargest = corrlab.findNstElem(data, target, k = data.shape[1])
    #     corrlab.saveFigCsv(klargest,period = period, output_dir=outputdir+'aggravated/klargest/', date=day,fontsize = 10)
    #     appointed = corr.appointedLst(data,analst)
    #     corrlab.saveFigCsv(appointed, period = period, output_dir=outputdir+'appointed/', date=day,fontsize = 20)
    # print day, 'Done!'

    entire_1s.sort_index()
    entire_5s.sort_index()
    entire_10s.sort_index()
    if type == 1:
        if not os.path.exists(u'/home/charles/python/intern/output/corr of target/'+target+'/aggravated/'):
            os.makedirs(u'/home/charles/python/intern/output/corr of target/'+target+'/aggravated/')
        entire_1s.to_csv(u'/home/charles/python/intern/output/corr of target/'+target+'/aggravated/ru_1s.csv')
        entire_5s.to_csv(u'/home/charles/python/intern/output/corr of target/'+target+'/aggravated/ru_5s.csv')
        entire_10s.to_csv(u'/home/charles/python/intern/output/corr of target/'+target+'/aggravated/ru_10s.csv')
    elif type == 0:
        if not os.path.exists(u'/home/charles/python/intern/output/corr of target/'+target+'/rolling/'):
            os.makedirs(u'/home/charles/python/intern/output/corr of target/'+target+'/rolling/')
        entire_1s.to_csv(u'/home/charles/python/intern/output/corr of target/'+target+'/rolling/ru_1s.csv')
        entire_5s.to_csv(u'/home/charles/python/intern/output/corr of target/'+target+'/rolling/ru_5s.csv')
        entire_10s.to_csv(u'/home/charles/python/intern/output/corr of target/'+target+'/rolling/ru_10s.csv')
