
import numpy as np
import theano
import theano.tensor as T

import cgml.types
from cgml.graph import ComputationalGraph
from cgml.layers import Layer
from cgml.layers.base import _make_shared

from nose.tools import assert_almost_equals

def assertModelWeightsMatch(model):

    x2 = T.fmatrix('x2')
    x1 = T.fvector('x1')

    ravel2d = theano.function([x2],outputs=x2.ravel())
    ravel1d = theano.function([x1],outputs=x1.ravel())

    for layer,dropoutLayer in zip(model.layers,model.dropoutLayers):
        
        W,b = layer.weights()
        W_prime,b_prime = dropoutLayer.weights()
        
        q = 1 - dropoutLayer.dropout

        assert_almost_equals(sum(ravel2d(W)), sum(q * ravel2d(W_prime))) #< 1e-5:
        #print ravel2d(W),ravel2d(W_prime)
        #assert False

        assert_almost_equals(sum(ravel1d(b)), sum(ravel1d(b_prime))) #< 1e-5:
        #print ravel1d(b),ravel1d(b_prime)
        #assert False
        
def test_layers():

    schema = {'description':'test CG',
              'type':'classification',
              'supervised-cost': {'type': 'negative-log-likelihood',
                                  'name': 'class-out'},
              'graph':
                  [{'activation':'linear',
                    'n_in':100,
                    'n_out':10,
                    'dropout':0.5,
                    'name':'hidden1'},
                   {'activation':'sigmoid',
                    'n_in':10,
                    'n_out':3,
                    'dropout':0.0,
                    'name':'class-out'}]}
     
    model = ComputationalGraph(schema = schema,
                               seed = 0)

    assertModelWeightsMatch(model)

    y_train = np.asarray([0]).astype(cgml.types.intX)
    x_train = np.asarray([range(100)]).astype(cgml.types.floatX)

    model.setTrainDataOnDevice(x_train,y_train)
    
    model.supervised_update(0,1)

    assertModelWeightsMatch(model)


def test_varyingDropoutRates():

    X = T.fmatrix('X')

    rng = np.random.RandomState(seed=0)

    W = _make_shared(np.ones((10,1),dtype=cgml.types.floatX))
    b = _make_shared(np.zeros((1,), dtype=cgml.types.floatX))

    for dropout in [0.0,0.5,1.0]:
        
        layer = Layer(rng=rng,
                      input=X,
                      n_in=10,
                      n_out=1,
                      activation=cgml.activations.linear,
                      W=W,
                      b=b,
                      dropout=dropout,
                      name="dropout")
    
        f = theano.function(inputs=[X],
                            outputs=layer.output)
        
        X_np = np.ones((1,10),dtype=cgml.types.floatX)

        values = []

        for i in xrange(10000):
        
            y_np = f(X_np)
            
            values.append(y_np[0,0])

        assert_almost_equals(np.mean(values),(1-dropout)*10,places=1)
        
    
