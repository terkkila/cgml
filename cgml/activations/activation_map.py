
import theano.tensor as T
from cgml.activations import linrect

activationMap = {'linear':  None,
                 'sigmoid': T.nnet.sigmoid,
                 'tanh':    T.tanh,
                 'softmax': T.nnet.softmax,
                 'linrect': linrect,
                 'conv2d': T.nnet.conv2d}


