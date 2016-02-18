
import theano
from theano.tensor.signal.downsample import max_pool_2d

from .base import _dropout_from_layer_input
from .base import _make_images_from_vectors
from .base import _make_convolution_filters
from .base import _make_shared

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
                 filter_width = 0,
                 subsample = None,
                 maxpool = None,
                 name = 'unnamed'):

        self.name = name
        
        self.dropout = dropout

        if input.ndim == 2:
            input = _make_images_from_vectors(input,size=(n_in[1],n_in[2]))

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

        self.filter_width = filter_width
        self.subsample = subsample
        self.maxpool = maxpool

        # Create W if not given
        if not W:
            self.W = _make_shared( _make_convolution_filters(rng = rng,
                                                             n_filters_in = n_in[0],
                                                             n_filters_out = n_out[0],
                                                             filter_width = filter_width,
                                                             randomInit = randomInit),
                                   name = 'W' + '_' + self.name)
        else:
            self.W = W
        

        assert self.input.ndim == 4

        # Output of the 2D convolution is a 4D tensor
        self.output_pre = self.activation(self.input,
                                          self.W,
                                          subsample = self.subsample )
        
        self.output = max_pool_2d(self.output_pre, 
                                  self.maxpool, 
                                  ignore_border = True)

        self.params = [self.W]

        self.weights = theano.function( inputs = [],
                                        outputs = self.W )

        print(self.name + ": " + str(self.n_in) + ' ' + str(self.n_out) + ' ' + \
            str(self.input.ndim) + ' ' + str(self.output.ndim))
        
