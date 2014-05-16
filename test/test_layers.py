
import numpy as np
import theano
import theano.tensor as T
from cgml.computational_graph import ComputationalGraph

def assertModelWeightsMatch(model):
    
    for layer,dropoutLayer in zip(model.layers,model.dropoutLayers):
        
        W,b = layer.weights()
        W_prime,b_prime = dropoutLayer.weights()
        
        q = 1 - dropoutLayer.p

        assert sum(W.flatten() - q * W_prime.flatten()) == 0
        assert sum(b.flatten() - b_prime.flatten()) == 0
        
def test_layers():

    rng = np.random.RandomState(0)

    schema = {'description':'test CG',
              'type': 'classifier',
              'randomInit':True,
              'graph':
                  [{'activation':'linear',
                    'n_in':100,
                    'n_out':10,
                    'dropout':0.5},
                   {'activation':'sigmoid',
                    'n_in':10,
                    'n_out':3,
                    'dropout':0.0}]}
     
    model = ComputationalGraph(schema = schema,
                               learnRate = 0.01,
                               momentum = 0.0,
                               seed = 0)

    assertModelWeightsMatch(model)

    y_train = np.asarray([0])
    x_train = np.asarray([range(100)])
    
    model.supervised_update(x_train,y_train)

    assertModelWeightsMatch(model)

    
    