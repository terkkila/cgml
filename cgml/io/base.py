
import numpy as np
import theano
import json

def _read_data(fileName):

    yx = np.loadtxt(fileName, delimiter = '\t')
    
    x = np.take(yx,range(1,yx.shape[1]),axis=1)
    y = yx[range(yx.shape[0]),0]

    return x,y


def ppf(x):
    if x < 0:
        return '{0:.3f}'.format(x)
    return ' {0:.3f}'.format(x)
    











