
import numpy as np
import theano
import theano.tensor as T
from cgml.computational_graph import ComputationalGraph

def assertModelWeightsMatch(model):

    x2 = T.fmatrix('x2')
    x1 = T.fvector('x1')

    ravel2d = theano.function([x2],outputs=x2.ravel())
    ravel1d = theano.function([x1],outputs=x1.ravel())

    for layer,dropoutLayer in zip(model.layers,model.dropoutLayers):
        
        W,b = layer.weights()
        W_prime,b_prime = dropoutLayer.weights()
        
        q = 1 - dropoutLayer.dropout

        if not sum(ravel2d(W) - q * ravel2d(W_prime)) < 1e-5:
            print ravel2d(W),ravel2d(W_prime)
            assert False

        if not sum(ravel1d(b) - ravel1d(b_prime)) < 1e-5:
            print ravel1d(b),ravel1d(b_prime)
            assert False
        
def test_layers():

    rng = np.random.RandomState(0)

    schema = {'description':'test CG',
              'supervised-cost': {'type': 'negative-log-likelihood',
                                  'name': 'class-out'},
              'graph':
                  [{'activation':'linear',
                    'n_in':100,
                    'n_out':10,
                    'dropout':0.5},
                   {'activation':'sigmoid',
                    'n_in':10,
                    'n_out':3,
                    'dropout':0.0,
                    'name':'class-out'}]}
     
    model = ComputationalGraph(schema = schema,
                               learnRate = 0.01,
                               momentum = 0.0,
                               seed = 0)

    assertModelWeightsMatch(model)

    y_train = np.asarray([0]).astype(np.int32)
    x_train = np.asarray([range(100)]).astype(theano.config.floatX)
    
    model.supervised_update(x_train,y_train)

    assertModelWeightsMatch(model)

    
    
