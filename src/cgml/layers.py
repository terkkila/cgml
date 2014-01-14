
import theano.tensor as T
from data import makeRandomSharedLayerParams
        
class Layer(object):

    def __init__(self, input = None, n_in = None, n_out = None, activation = None):

        self.input = input

        self.W,self.b = makeRandomSharedLayerParams(n_in       = n_in,
                                                    n_out      = n_out,
                                                    activation = activation)
        
        self.output = activation(T.dot(input,self.W) + self.b)

        self.params = [self.W,self.b]


        

        


        











