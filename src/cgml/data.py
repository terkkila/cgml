
import numpy as np
import theano
import theano.tensor as T

def makeRandomClassificationData(n = None, n_in = None, n_out = None):

    x_train = np.random.rand(n,n_in)
    y_train = np.random.randint(0,n_out,n)

    return (x_train,y_train)

def makeRandomRegressionData(n = None, n_in = None, n_out = None):

    x_train = np.random.rand(n,n_in)
    y_train = np.random.rand(n,1)

    return (x_train,y_train)
    
    
def makeShared(data, borrow=True):
    """ Function that loads the dataset into shared variables
    
    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """

    return theano.shared(np.asarray(data,
                                    dtype = theano.config.floatX),
                         borrow = borrow)


def makeSharedWeightMatrix(rng        = None,
                           n_in       = None,
                           n_out      = None,
                           activation = None,
                           randomInit = True):
    
    # `W` is initialized with `W_values` which is uniformely sampled
    # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
    # for tanh activation function
    # the output of uniform is converted using asarray to dtype
    # theano.config.floatX so that the code is runable on GPU
    # Note : optimal initialization of weights is dependent on the
    #        activation function used (among other things).
    #        For example, results presented in [Xavier10]_ suggest that you
    #        should use 4 times larger initial weights for sigmoid
    #        compared to tanh
    #        We have no info for other function, so we use the same as tanh.
    if randomInit:
        W_values = np.asarray(rng.uniform(low  = -np.sqrt(6. / (n_in + n_out)),
                                          high = np.sqrt(6. / (n_in + n_out)),
                                          size = (n_in, n_out) ),
                              dtype = theano.config.floatX)
        
        # If the activation function happens to be sigmoid,
        # we multiply the weights by 4
        if activation == T.nnet.sigmoid:
            W_values *= 4

    # In case we do not want random weights, we set all of them to zero
    else:
        W_values = np.zeros((n_in,n_out), dtype = theano.config.floatX)

    # Make W shared theano variable
    W = theano.shared(value = W_values, name = 'W')
    
    return W


def makeSharedBiasVector(rng        = None,
                         n_in       = None,
                         n_out      = None,
                         activation = None,
                         randomInit = True):
    

    if randomInit:

        b_values = np.asarray(rng.uniform(low = -1,
                                          high = 1,
                                          size = (n_out,)),
                              dtype = theano.config.floatX)
    
    else:

        b_values = np.zeros((n_out,), dtype = theano.config.floatX)

    # ... and make b shared theano variable
    b = theano.shared(value = b_values, name='b')

    return b










