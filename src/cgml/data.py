
import numpy as np
import theano
import theano.tensor as T

def makeRandomClassificationData(n = None, n_in = None, n_out = None):

    x_train = np.random.rand(n,n_in)
    y_train = np.random.randint(0,n_out,n)

    return (x_train,y_train)

    
def shareData(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables
    
    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy

    shared_x = theano.shared(np.asarray(data_x,
                                        dtype = theano.config.floatX),
                             borrow = borrow)
    
    shared_y = theano.shared(np.asarray(data_y,
                                        dtype = theano.config.floatX),
                             borrow = borrow)

    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')


def makeRandomSharedLayerParams(n_in = None,
                          n_out      = None,
                          activation = None):
    
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
    W_values = np.asarray(np.random.uniform(low  = -np.sqrt(6. / (n_in + n_out)),
                                            high = np.sqrt(6. / (n_in + n_out)),
                                            size = (n_in, n_out) ),
                          dtype = theano.config.floatX)

    if activation == T.nnet.sigmoid:
        W_values *= 4
        
    W = theano.shared(value = W_values, name = 'W')
        
    b_values = np.zeros((n_out,), dtype = theano.config.floatX)
    b = theano.shared(value = b_values, name='b')

    return W,b









