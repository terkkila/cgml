
import theano
import theano.tensor as T
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

        self.input = input
        self.n_in  = n_in
        self.n_out = n_out

        # If parameters for the layer are not given
        if not W and not b:

            # We create new shared layer parameters
            self.W,self.b = makeSharedLayerParams(rng        = rng,
                                                  n_in       = n_in,
                                                  n_out      = n_out,
                                                  activation = activation,
                                                  randomInit = randomInit)
        else:

            # Otherwise we take the given parameters
            self.W = W
            self.b = b
            
        # If activation function is defined, use it,
        # otherwise assign linear activation
        if activation != None:
            self.output = activation(T.dot(input,self.W) + self.b)
        else:
            self.output = T.dot(input,self.W) + self.b
            
        self.params = [self.W,self.b]

        
def _dropout_from_layer_output(rng, output, p):
    """p is the probablity of dropping a unit
    """
    srng = T.shared_randomstreams.RandomStream(rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n    = 1,
                         p    = 1 - p,
                         size = output.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    return output * T.cast(mask, theano.config.floatX)
    

class DropoutLayer(Layer):

    def __init__(self,
                 rng,
                 input = None,
                 n_in = None,
                 n_out = None,
                 activation = None,
                 randomInit = True,
                 W = None,
                 b = None):
        
        super(DropoutLayer, self).__init__(
            rng = rng,
            input = input,
            n_in = n_in,
            n_out = n_out,
            W = W,
            b = b,
            randomInit = randomInit,
            activation = activation)

        self.output = _dropout_from_layer_output(rng,
                                                 self.output,
                                                 p = 0.5)
        

        


        











