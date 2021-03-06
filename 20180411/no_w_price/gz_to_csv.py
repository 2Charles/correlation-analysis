import os
import pandas as pd
class Gz2Csv(object):
    def __init__(self, dir):
        self.dir = dir
        self.generated_file = []
        self.generated_folder = []

    def gz2csv(self, dir='default'):
        if dir == 'default':
            dir = self.dir
        for fname in os.listdir(dir):
            name = fname.split('.')[0]
            tmp = pd.read_csv(dir+'/'+fname, header=0,index_col=0,compression='gzip')
            if not os.path.exists(dir+'/'+'/generated_csv/'):
                os.makedirs(dir+'generated_csv/')
                self.generated_folder.append(dir+'generated_csv/')
            tmp.to_csv(dir+'generated_csv/'+name+'.csv')
            self.generated_file.append(dir+'generated_csv/'+name+'.csv')

    def csv2gz(self, dir='default'):
        if dir == 'default':
            dir = self.dir
        for fname in os.listdir(dir):
            name = fname.split('.')[0]
            tmp = pd.read_csv(dir+'/'+fname, header=0,index_col=0)
            if not os.path.exists(dir+'generated_gz/'):
                os.makedirs(dir+'generated_gz/')
                self.generated_folder.append(dir+'generated_gz/')
            tmp.to_csv(dir+'generated_gz/'+name+'.dat.gz',compression='gzip')
            self.generated_file.append(dir+'generated_gz/'+name+'.dat.gz')

    def delete_generated_file(self):
        for fname in self.generated_file:
            if os.path.exists(fname):
                os.remove(fname)
        for folder in self.generated_folder:
            if os.path.exists(folder):
                os.rmdir(folder)
        self.generated_file=[]
        self.generated_folder=[]
        print 'generated files and folders deleted!'
