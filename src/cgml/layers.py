
import theano
import theano.tensor as T
import theano.tensor.shared_randomstreams
from data import makeSharedWeightMatrix,makeSharedBiasVector

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
    dropoutInput = input * T.cast(mask, theano.config.floatX)
    return dropoutInput

        
class Layer(object):

    def __init__(self,
                 rng = None,
                 input = None,
                 n_in = None,
                 n_out = None,
                 activation = None,
                 randomInit = True,
                 W = None,
                 b = None,
                 dropout = 0):

        self.rng = rng

        self.dropout = dropout

        # If dropout is positive, apply it to the input
        if dropout > 0:

            self.input = _dropout_from_layer_input(rng   = self.rng,
                                                   input = input,
                                                   p     = self.dropout)
        else:

            self.input = input

        self.n_in  = n_in
        self.n_out = n_out
        self.activation = activation

        # Create W if not given
        if not W:
            self.W = makeSharedWeightMatrix(rng = rng,
                                            n_in = n_in,
                                            n_out = n_out,
                                            activation = self.activation,
                                            randomInit = randomInit)
        else:
            self.W = W
        
            
        # Create b if not given
        if not b:
            self.b = makeSharedBiasVector(rng = rng,
                                          n_in = n_in,
                                          n_out = n_out,
                                          activation = self.activation,
                                          randomInit = False)
        else:
            self.b = b

        # If activation function is defined, use it,
        # otherwise assign linear activation
        if self.activation != None:
            self.output = self.activation(T.dot(input,self.W) + self.b)
        else:
            self.output = T.dot(input,self.W) + self.b
            
        self.params = [self.W,self.b]

        self.weights = theano.function( inputs = [],
                                        outputs = (self.W,self.b) )

class ConvolutionLayer(object):

    def __init__(self,
                 rng = None,
                 input = None,
                 n_in = None,
                 n_out = None,
                 activation = None,
                 randomInit = True,
                 W = None,
                 dropout = 0,
                 n_filters = None,
                 filter_dim = 0):

        assert n_filters == 1

        self.rng = rng

        self.dropout = dropout

        # If dropout is positive, apply it to the input
        if dropout > 0:

            self.input = _dropout_from_layer_input(rng   = self.rng,
                                                   input = input,
                                                   p     = self.dropout)
        else:

            self.input = input

        self.n_in  = n_in
        self.n_out = n_out
        self.activation = activation

        self.n_filters = n_filters
        self.filter_dim = filter_dim

        # Create W if not given
        if not W:
            self.W = makeSharedWeightMatrix(rng = rng,
                                            n_in = filter_dim,
                                            n_out = filter_dim,
                                            activation = self.activation,
                                            randomInit = randomInit)
        else:
            self.W = W
        
        self.output = self.input
            
        #self.output = self.activation(self.input,self.W)

        self.params = [self.W]

        self.weights = theano.function( inputs = [],
                                        outputs = self.W )


        

        


        











