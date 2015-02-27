
import numpy as np
import theano
import theano.tensor as T
#import theano.tensor.shared_randomstreams


def _dropout_from_layer_input(input = None,
                              p     = None,
                              rng   = None):
    """p is the probablity of dropping a unit
    """
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n    = 1,
                         p    = 1 - p,
                         size = input.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    dropoutInput = T.cast(input * mask, theano.config.floatX)
    return dropoutInput

        
def _make_images_from_vectors(x,size=None):

    return T.reshape(x,(x.shape[0],
                        1,
                        size[0],
                        size[1]))


def _make_vectors_from_images(x):

    return T.reshape(x,(x.shape[0],
                        T.prod(x.shape[1:])))

    
def _make_shared(data, borrow = True, name = None):
    """ Function that loads the dataset into shared variables
    
    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """

    return theano.shared(np.asarray(data),
                         borrow = borrow,
                         name = name)

    
def _make_weight_matrix(rng        = None,
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
    
    return W_values

def _make_convolution_filters(rng = None,
                              n_filters_in = None,
                              n_filters_out = None,
                              filter_width = None,
                              randomInit = True):

    W_values = []
    
    for i in xrange(n_filters_in*n_filters_out):
        W_values.append( _make_weight_matrix(rng = rng,
                                             n_in = filter_width[0],
                                             n_out = filter_width[1],
                                             randomInit = randomInit) )
    
    return np.asarray(W_values).reshape((n_filters_out,
                                         n_filters_in,
                                         filter_width[0],
                                         filter_width[1]))


def _make_bias_vector(rng        = None,
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

    return b_values





        


        











