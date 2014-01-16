
import theano.tensor as T
from data import makeRandomSharedLayerParams
        
class Layer(object):

    def __init__(self, input = None, n_in = None, n_out = None, activation = None, W = None, b = None):

        self.input = input

        if W == None and b == None:
            self.W,self.b = makeRandomSharedLayerParams(n_in       = n_in,
                                                        n_out      = n_out,
                                                        activation = activation)
        else:
            self.W = W
            self.b = b

            
        # If activation function is defined, use it, otherwise assign linear activation
        if activation != None:
            self.output = activation(T.dot(input,self.W) + self.b)
        else:
            self.output = T.dot(input,self.W) + self.b
            
        self.params = [self.W,self.b]


        

        


        











