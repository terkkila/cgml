
import numpy as np

def readData(fileName):

    yx = np.loadtxt(fileName, delimiter = '\t')
    
    x = np.take(yx,range(1,yx.shape[1]),axis=1)
    y = yx[range(yx.shape[0]),0]

    return x,y
    
class DataReader(object):

    def __init__(self,
                 fileName, 
                 batchSize = 1):

        self.batchSize = batchSize

        self.f = open(fileName,'r')

        self.nCols = len(self.f.readline().rstrip().split('\t'))

        self.f.seek(0)

    def __iter__(self):
        return self

    def next(self):

        lines = []
        nLinesRead = 0

        for line in self.f:
            lines.append( line.rstrip() )
            nLinesRead += 1
            if nLinesRead == self.batchSize:
                break

        if nLinesRead == 0:
            raise StopIteration

        
        return np.asarray([map(float,line.split('\t')) for line in lines],
                          dtype = float)


    def rewind(self):
        self.f.seek(0)












