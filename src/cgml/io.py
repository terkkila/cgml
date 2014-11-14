
import numpy as np
import theano

def readData(fileName):

    yx = np.loadtxt(fileName, delimiter = '\t')
    
    x = np.take(yx,range(1,yx.shape[1]),axis=1)
    y = yx[range(yx.shape[0]),0]

    return x,y

def ppf(x):
    if x < 0:
        return '{0:.3f}'.format(x)
    return ' {0:.3f}'.format(x)
    
class DataReader(object):

    def __init__(self,
                 fileName, 
                 batchSize = 1,
                 targetType = theano.config.floatX,
                 delimiter = '\t'):

        self.batchSize = batchSize

        self.delimiter = delimiter

        self.f = open(fileName,'r')

        self.nCols = len(self.f.readline().rstrip().split(self.delimiter))

        self.f.seek(0)

        self.targetType = targetType

    def __iter__(self):
        return self


    def readBatch(self,batchSize):

        lines = []

        for line in self.f:
            lines.append( line.rstrip() )
            if len(lines) == batchSize:
                break

        return lines


    def parseBatch(self,lines):

        arr =  np.asarray([map(float,line.split(self.delimiter)) for line in lines],
                          dtype = float)
        
        y = np.asarray(arr.take(0,axis=1),dtype=self.targetType)
        x = arr.take(xrange(1,arr.shape[1]),axis=1).astype(theano.config.floatX)

        return x,y


    def next(self):

        lines = self.readBatch(self.batchSize)

        if len(lines) == 0:
            raise StopIteration

        x,y = self.parseBatch(lines)

        return x,y

    def cache(self):
        self.rewind()
        return self.parseBatch(self.readBatch(-1))

    def rewind(self):
        self.f.seek(0)












