
import theano
import theano.tensor as T
import numpy as np
from cgml.computational_graph import ComputationalGraph

def test_conv2d():

    x2_sym = T.dmatrix('x2')
    filters_sym = T.tensor4('filters_sym')

    f = theano.function( inputs = [x2_sym,filters_sym],
                         outputs = T.nnet.conv2d(T.reshape(x2_sym,
                                                           (x2_sym.shape[0],
                                                            1,
                                                            T.cast(T.sqrt(x2_sym.shape[1]),'int32'),
                                                            T.cast(T.sqrt(x2_sym.shape[1]),'int32'))),
                                                 filters_sym,
                                                 border_mode = 'valid'))

    X = np.asarray([[1,2,3,4,5,6,7,8,9],[10,11,12,13,14,15,16,17,18]])

    filters = np.asarray([[[0,0,0],[0,1,0],[0,0,0]],
                          [[0,0,0],[0,2,0],[0,0,0]]]).reshape((2,1,3,3))
    

    Y = f(X,filters)

    assert Y.shape == (2,2,1,1)
    assert Y[0][0][0][0] == 5
    assert Y[1][0][0][0] == 14
    assert Y[0][1][0][0] == 10
    assert Y[1][1][0][0] == 28

def test_conv2d_layer():

    rng = np.random.RandomState(0)

    schema = {'description':'test CG',
              'type': 'classifier',
              'randomInit':True,
              'graph':
                  [{'activation':'conv2d',
                    'n_filters':2,
                    'filter_width':3,
                    'subsample':[1,1],
                    'n_in':16,
                    'n_out':8,
                    'dropout':0.0},
                   {'activation':'sigmoid',
                    'n_in':8,
                    'n_out':3,
                    'dropout':0.0}]}
     
    model = ComputationalGraph(schema = schema,
                               learnRate = 0.01,
                               momentum = 0.0,
                               seed = 0)
