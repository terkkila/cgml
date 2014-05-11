
import theano.tensor as T


def nllCost(yhat, y):
    """Calculates the negative log-likelihood between
    the class probabilities and true class labels.
    """
    
    # note on syntax: T.arange(y.shape[0]) is a vector of integers [0,1,2,...,len(y)-1].
    # Indexing a matrix M by the two vectors [0,1,...,K], [a,b,...,k] returns the
    # elements M[0,a], M[1,b], ..., M[K,k] as a vector.  Here, we use this
    # syntax to retrieve the log-probability of the correct labels, y.
    return -T.mean(T.log(yhat)[T.arange(y.shape[0]),y])

    
def sqerrCost(yhat, y):

    # Error
    e = y - yhat

    # Squared error
    return( T.sum(T.dot(e,e.T)) )
