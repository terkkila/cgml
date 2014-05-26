
import theano.tensor as T
import numpy as np

def nllCost(yhat, y):
    """Calculates the negative log-likelihood between
    the class probabilities and true class labels.
    """
    
    # note on syntax: T.arange(y.shape[0]) is a vector of integers [0,1,2,...,len(y)-1].
    # Indexing a matrix M by the two vectors [0,1,...,K], [a,b,...,k] returns the
    # elements M[0,a], M[1,b], ..., M[K,k] as a vector.  Here, we use this
    # syntax to retrieve the log-probability of the correct labels, y.
    return -T.mean(T.log(yhat)[T.arange(y.shape[0]),y])

    
def sqerrCost_old(yhat, y):

    # Error
    e = y - yhat

    # Squared error
    return( T.mean(T.diagonal(T.dot(e,e.T))) )

def sqerrCost(yhat, y):

    # Error
    e = y - yhat

    # Mean of MSEs over 2
    return( T.mean(T.mean(e*e)) / 2 )


def crossEntCost(yhat,y):
    return T.mean(-T.mean(y * T.log(yhat) + (1 - y) * T.log(1 - yhat), axis=1))

costMap = {
    'negative-log-likelihood': nllCost,
    'squared-error': sqerrCost,
    'cross-entropy': crossEntCost
    }
