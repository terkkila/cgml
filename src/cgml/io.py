
import numpy as np

def readData(fileName):

    yx = np.loadtxt(fileName, delimiter = '\t')
    
    x = np.take(yx,range(1,yx.shape[1]),axis=1)
    y = yx[range(yx.shape[0]),0]

    return x,y
    
class DataReader(object):

    def __init__(self,fileName):
        self.f = open(fileName,'r')
        self.nInputs = len(self.f.readline().rstrip().split('\t')[1:])
        self.f.seek(0)

    def __iter__(self):
        return self

    def next(self):

        try:
            
            line = self.f.readline().rstrip()
            y = np.asarray([int(float(line.split('\t')[0]))], dtype = int).reshape((1,))
            x = np.asarray(map(float,line.split('\t')[1:]), dtype = float).reshape((1,self.nInputs))
            return x,y
            
        except:
            raise StopIteration

    def rewind(self):
        self.f.seek(0)












