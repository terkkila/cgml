
import theano.tensor as T
import numpy as np
from theano.ifelse import ifelse

def negativeLogLikelihood(yhat, y):
    """Calculates the negative log-likelihood between
    the class probabilities and true class labels.
    """
    
    # note on syntax: T.arange(y.shape[0]) is a vector of integers [0,1,2,...,len(y)-1].
    # Indexing a matrix M by the two vectors [0,1,...,K], [a,b,...,k] returns the
    # elements M[0,a], M[1,b], ..., M[K,k] as a vector.  Here, we use this
    # syntax to retrieve the log-probability of the correct labels, y.
    return -T.mean(T.log(yhat)[T.arange(y.shape[0]),y])

def squaredError(yhat, y):

    # Error
    e = y - yhat

    # Mean of MSEs over 2
    return( T.mean(T.mean(e*e)) )


def crossEntropy(yhat,y):
    return T.mean(-T.mean(y * T.log(yhat) + (1 - y) * T.log(1 - yhat), axis=1))


def absoluteError(yhat,y):

    e = y - yhat

    return T.mean(T.abs_(e.ravel()))

def absolutePercentageError(yhat,y):

    e = (y - yhat) / y

    return T.mean(T.abs_(e.ravel()))

def huberError(yhat,y):

    delta = 1/2

    e = y - yhat

    a = .5 * e**2

    b = delta * (abs(e) - delta / 2.)

    l = T.switch(abs(e) <= delta, a, b)

    return l.sum()

