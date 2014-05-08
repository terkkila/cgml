
import numpy as np

def readData(fileName):

    yx = np.loadtxt(fileName, delimiter = '\t')
    
    x = np.take(yx,range(1,yx.shape[1]),axis=1)
    y = yx[range(yx.shape[0]),0]

    return x,y
    
class DataReader(object):

    def __init__(self,fileName):
        self.f = open(fileName,'r')
        self.nCols = len(self.f.readline().rstrip().split('\t'))
        self.f.seek(0)

    def __iter__(self):
        return self

    def next(self):

        try:
            
            return np.asarray(map(float,self.f.readline().rstrip().split('\t')), 
                              dtype = float).reshape((1,self.nCols))
            
        except:
            raise StopIteration

    def rewind(self):
        self.f.seek(0)












