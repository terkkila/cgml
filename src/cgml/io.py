
import numpy as np
import theano
import json

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
                 targetType = theano.config.floatX):

        self.batchSize = batchSize

        self.f = open(fileName,'r')

        self.f.seek(0)

        self.targetType = targetType

    def __iter__(self):
        return self


    def readBatch(self,batchSize):

        lines = []

        for line in self.f:
            lines.append( json.loads(line.rstrip()) )
            if len(lines) == batchSize:
                break

        return lines


    def parseBatch(self,lines):

        y = np.asarray([obj['y'] for obj in lines],dtype=self.targetType)
        x = np.asarray([obj['x'] for obj in lines],dtype=theano.config.floatX)
        sampleIDs = [obj['sampleID'] for obj in lines]

        #arr =  np.asarray([map(float,line.split(self.delimiter)) for line in lines],
        #                  dtype = float)
        
        #y = np.asarray(arr.take(0,axis=1),dtype=self.targetType)
        #x = arr.take(xrange(1,arr.shape[1]),axis=1).astype(theano.config.floatX)

        return sampleIDs,x,y


    def next(self):

        lines = self.readBatch(self.batchSize)

        if len(lines) == 0:
            raise StopIteration

        return self.parseBatch(lines)


    def cache(self):
        self.rewind()
        return self.parseBatch(self.readBatch(-1))


    def rewind(self):
        self.f.seek(0)












