
import theano
import theano.tensor as T
import numpy as np

import cgml.types
from cgml.activations import conv2d
from cgml.graph import ComputationalGraph
from cgml.layers.base import _make_images_from_vectors
from cgml.layers.base import _make_convolution_filters
from cgml.layers import ConvolutionLayer

def test_conv2d():

    x2_sym = T.fmatrix('x2')
    filters_sym = T.tensor4('filters1_sym')

    f_sym = conv2d( _make_images_from_vectors(x2_sym,size=(3,3)),
                    filters_sym,
                    border_mode = 'valid')

    
    f = theano.function( inputs  = [x2_sym,filters_sym],
                         outputs = f_sym)

    X = np.asarray([[1,2,3,4,5,6,7,8,9],[10,11,12,13,14,15,16,17,18]]).astype(cgml.types.floatX)

    filters = np.asarray([[[0,0,0],[0,1,0],[0,0,0]],
                          [[0,0,0],[0,2,0],[0,0,0]]]).reshape((2,1,3,3)).astype(cgml.types.floatX)
    

    Y = f(X,filters)

    assert Y.shape == (2,2,1,1)
    assert Y[0][0][0][0] == 5
    assert Y[1][0][0][0] == 14
    assert Y[0][1][0][0] == 10
    assert Y[1][1][0][0] == 28


def test_conv2d_layers():

    rng = np.random.RandomState(1234)

    x_sym = T.fmatrix('x_sym')

    cl1 = ConvolutionLayer(input = x_sym,
                           activation = T.nnet.conv2d,
                           rng = rng,
                           filter_width = [2,2],
                           subsample = [1,1],
                           maxpool = [1,1],
                           n_in = [1,4,4],
                           n_out = [2,3,3],
                           dropout = 0.0,
                           name = 'cl1')

    cl2 = ConvolutionLayer(input = cl1.output,
                           activation = T.nnet.conv2d,
                           rng = rng,
                           filter_width = [3,3],
                           subsample = [1,1],
                           maxpool = [1,1],
                           n_in = [2,3,3],
                           n_out = [2,1,1],
                           dropout = 0.0,
                           name = 'cl2')

    out1 = theano.function(inputs = [x_sym],
                           outputs = cl1.output)
    
    out2 = theano.function(inputs = [x_sym],
                           outputs = cl2.output)

    x = np.asarray([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]).astype(cgml.types.floatX)

    print(out1(x).shape)
    assert out1(x).shape == (1,2,3,3)

    print(out2(x).shape)
    assert out2(x).shape == (1,2,1,1)

    
def test_conv2d_graph():

    schema = {'description':'test CG',
              'type':'classification',
              'supervised-cost': {'type': 'negative-log-likelihood', 
                                  'name': 'class-out'},
              'graph':
                  [{'activation':'conv2d',
                    'filter_width':[2,2],
                    'subsample':[1,1],
                    'maxpool':[1,1],
                    'n_in':[1,4,4],
                    'n_out':[2,3,3],
                    'dropout':0.0,
                    'name':'conv1'},
                   {'activation':'conv2d',
                    'filter_width':[3,3],
                    'subsample':[1,1],
                    'maxpool':[1,1],
                    'n_in':[2,3,3],
                    'n_out':[2,1,1],
                    'dropout':0.0,
                    'name':'conv2'},
                   {'activation':'sigmoid',
                    'n_in':2,
                    'n_out':2,
                    'dropout':0.0,
                    'name':'class-out'}]}
     
    model = ComputationalGraph(schema = schema,
                               seed = 0)

    X = np.asarray([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]).astype(cgml.types.floatX)

    y_hat = model.predict(X)

    y = np.asarray([0]).astype(cgml.types.intX)

    model.setTrainDataOnDevice(X,y)

    model.supervised_update(0,1)


def test_conv2d_graph2():

   schema = {
       'description':'foo',
       'type':'classification',
       'supervised-cost':{
           'type':'negative-log-likelihood',
           'name':'class-out'
           },
       'graph': [{
               'activation': 'conv2d',
               'n_in':         [1,10,14],
               'filter_width': [5,5],
               'subsample':    [2,2],
               'maxpool':      [1,1],
               'n_out':        [10,3,5],
               'dropout':      0.2,
               'name':         'conv1'
               },{
               'activation': 'sigmoid',
               'n_in':       150,
               'n_out':      2,
               'dropout':    0.2,
               'name':       'class-out'
               }]
       }

   model = ComputationalGraph(schema = schema,
                              seed = 0)

   x = np.random.uniform(size=(10,14)).reshape((1,10*14)).astype(cgml.types.floatX)
   y = np.asarray([0]).astype(cgml.types.intX)

   model.setTrainDataOnDevice(x,y)

   model.supervised_update(0,1)
