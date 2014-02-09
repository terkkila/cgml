
import theano.tensor as T
from data import makeSharedLayerParams
        
class Layer(object):

    def __init__(self,
                 input = None,
                 n_in = None,
                 n_out = None,
                 activation = None,
                 randomInit = True):

        self.input = input
        self.n_in  = n_in
        self.n_out = n_out
        
        self.W,self.b = makeSharedLayerParams(n_in       = n_in,
                                              n_out      = n_out,
                                              activation = activation,
                                              randomInit = randomInit)
            
        # If activation function is defined, use it, otherwise assign linear activation
        if activation != None:
            self.output = activation(T.dot(input,self.W) + self.b)
        else:
            self.output = T.dot(input,self.W) + self.b
            
        self.params = [self.W,self.b]


        

        


        











