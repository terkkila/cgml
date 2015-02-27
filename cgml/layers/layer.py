
import numpy as np
import theano
import theano.tensor as T

from .base import _make_vectors_from_images
from .base import _dropout_from_layer_input
from .base import _make_shared
from .base import _make_weight_matrix
from .base import _make_bias_vector

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
                 dropout = 0,
                 name = 'unnamed'):

        self.name = name

        self.dropout = dropout

        # If the layer is given an image
        if input.ndim == 4:
            input = _make_vectors_from_images(input)
            n_in = np.prod(n_in)

        # If dropout is positive, apply it to the input
        if dropout > 0:

            self.input = _dropout_from_layer_input(rng   = rng,
                                                   input = input,
                                                   p     = self.dropout)
        else:

            self.input = input

        self.n_in  = n_in
        self.n_out = n_out
        self.activation = activation

        # Create W if not given
        if not W:
            self.W = _make_shared( _make_weight_matrix(rng = rng,
                                                       n_in = n_in,
                                                       n_out = n_out,
                                                       activation = self.activation,
                                                       randomInit = randomInit),
                                   name = 'W' + '_' + self.name)
        else:
            self.W = W
        
            
        # Create b if not given
        if not b:
            self.b = _make_shared( _make_bias_vector(rng = rng,
                                                     n_in = n_in,
                                                     n_out = n_out,
                                                     activation = self.activation,
                                                     randomInit = False),
                                   name = 'b' + '_' + name)
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

        #print self.name + ": " + str(self.n_in) + ' ' + str(self.n_out) + ' ' + \
        #    str(self.input.ndim) + ' ' + str(self.output.ndim)
        

