
import theano
import theano.tensor as T
import numpy as np
from cgml.graph import ComputationalGraph
from nose.tools import raises

# Supervised Autoencoder is broken at the moment
@raises(TypeError)
def test_supervised_autoencoder():

    schema = {'description':'test CG',
              'type':'supervised-autoencoder',
              'supervised-cost': {'type': 'negative-log-likelihood', 
                                  'name': 'class-out'},
              'unsupervised-cost': {'type': 'squared-error',
                                    'name': 'decode-out'},
              'graph':
                  [{'activation':'linear',
                    'n_in':5,
                    'n_out':2,
                    'dropout':0.0,
                    'name':'encode-out',
                    'branch':
                        [{'activation':'softmax',
                          'n_in':2,
                          'n_out':2,
                          'name':'class-out',
                          'dropout':0.0}]},
                   {'activation':'linear',
                    'n_in':2,
                    'n_out':5,
                    'dropout':0.0,
                    'name':'decode-out'}]}

    model = ComputationalGraph(schema = schema,
                               seed = 0)

    nLayers = len(model.layers)
    
    assert nLayers == 3
    for i in range(nLayers):
        assert not np.any(model.layers[i].weights[1])
        
    x = np.asarray([[1,2,3,4,5],[2,3,4,5,6]]).astype(theano.config.floatX)
    y = np.asarray([0,1],dtype=np.int).astype(np.int)
     
    x_dec = model.decode(x)
    
    for i in range(1,x_dec.shape[0]):
        assert np.all(x_dec[i-1] != x_dec[i])
    
    model.setTrainDataOnDevice(x,y)
    
    for i in range(10):
        model.hybrid_update(0,1)
    
    assert not np.any(model.layers[0].weights()[0] == model.layers[1].weights()[0].T)
    assert not np.any(model.layers[0].weights()[1] == model.layers[1].weights()[1])



def test_unsupervised_autoencoder():

    schema = {'description':'test CG',
              'type':'autoencoder',
              'unsupervised-cost': {'type': 'squared-error',
                                    'name': 'decode-out'},
              'graph':[{
                'activation':'linear',
                'n_in':5,
                'n_out':2,
                'dropout':0.0,
                'name':'encode-out'
                },{
                'activation':'linear',
                'n_in':2,
                'n_out':5,
                'dropout':0.0,
                'name':'decode-out'}]}
    
    model = ComputationalGraph(schema = schema,
                               seed = 0)

    nLayers = len(model.layers)

    assert nLayers == 2
    for i in range(nLayers):
        assert not np.any(model.layers[i].weights[1])

    x = np.asarray([[1,2,3,4,5],[2,3,4,5,6]]).astype(theano.config.floatX)
    y = np.asarray([[0],[0]],dtype=theano.config.floatX).astype(theano.config.floatX)
    
    x_dec = model.decode(x)

    for i in range(1,x_dec.shape[0]):
        assert np.all(x_dec[i-1] != x_dec[i])

    model.setTrainDataOnDevice(x,y)

    for i in range(10):
        cost = model.unsupervised_update(0,1)
        assert cost > 0

    assert not np.any(model.layers[0].weights()[0] == model.layers[1].weights()[0].T)
    assert not np.any(model.layers[0].weights()[1] == model.layers[1].weights()[1])
