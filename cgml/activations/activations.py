
import theano.tensor as T
 
def linrect(x):
    return T.maximum(0,x)

def sigmoid(x):
    return T.nnet.sigmoid(x)

def tanh(x):
    return T.tanh(x)

def softmax(x):
    return T.nnet.softmax(x)

def conv2d(*args,**kwargs):
    return T.nnet.conv2d(*args,**kwargs)

def linear(x):
    return x
