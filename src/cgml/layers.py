
import theano
import theano.tensor as T
import theano.tensor.shared_randomstreams
from data import makeSharedLayerParams
        
class Layer(object):

    def __init__(self,
                 rng = None,
                 input = None,
                 n_in = None,
                 n_out = None,
                 activation = None,
                 randomInit = True,
                 W = None,
                 b = None):

        self.rng   = rng
        self.input = input
        self.n_in  = n_in
        self.n_out = n_out
        self.activation = activation
        
        # If parameters for the layer are not given
        if not W and not b:

            # We create new shared layer parameters
            self.W,self.b = makeSharedLayerParams(rng        = rng,
                                                  n_in       = n_in,
                                                  n_out      = n_out,
                                                  activation = self.activation,
                                                  randomInit = randomInit)
        else:

            # Otherwise we take the given parameters
            self.W = W
            self.b = b
            
        # If activation function is defined, use it,
        # otherwise assign linear activation
        if self.activation != None:
            self.output = self.activation(T.dot(input,self.W) + self.b)
        else:
            self.output = T.dot(input,self.W) + self.b
            
        self.params = [self.W,self.b]

        
def _dropout_from_layer_input(rng   = None,
                              input = None,
                              p     = None):
    """p is the probablity of dropping a unit
    """
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n    = 1,
                         p    = 1 - p,
                         size = input.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    return input * T.cast(mask, theano.config.floatX)
    

class DropoutLayer(Layer):

    def __init__(self,
                 rng = None,
                 input = None,
                 p = None,
                 *args,
                 **kwargs):
        
        Layer.__init__(self,
                       rng = rng,
                       input = input,
                       *args,
                       **kwargs)

        self.rng = rng
        self.input = input
        self.p = p

        self.input = _dropout_from_layer_input(rng   = self.rng,
                                               input = self.input,
                                               p     = self.p)
        

        


        











