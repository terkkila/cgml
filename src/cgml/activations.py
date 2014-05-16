
import theano.tensor as T

def linrect(x):
    return T.maximum(0,x)


activationMap = {'linear':  None,
                 'sigmoid': T.nnet.sigmoid,
                 'tanh':    T.tanh,
                 'softmax': T.nnet.softmax,
                 'linrect': linrect,
                 'conv2d': T.nnet.conv2d}
