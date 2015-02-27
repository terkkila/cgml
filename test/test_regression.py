
import theano
import theano.tensor as T
import numpy as np
from cgml.graph import ComputationalGraph

def test_linear_regression():

    schema = {'description':'test CG',
              'type': 'regression',
              'supervised-cost': {'type': 'squared-error', 
                                  'name': 'scalar-out'},
              'target-scaling': {'mean': 0, 'stdev': 1},
              'graph':
                  [{'activation':'linear',
                    'n_in':5,
                    'n_out':1,
                    'dropout':0.0,
                    'name':'scalar-out'}]}

    model = ComputationalGraph(schema = schema,
                               seed = 0)

    nLayers = len(model.layers)

    assert nLayers == 1
    for i in xrange(nLayers):
        assert not np.any(model.layers[i].weights[1])
        
    assert model.targetType == theano.config.floatX
        
    x = np.asarray([[1,2,3,4,5],[2,3,4,5,6]]).astype(theano.config.floatX)
    y = np.asarray([[0],[1]],dtype=theano.config.floatX).astype(theano.config.floatX)
    
    yhat = model.predict(x)

    print yhat.shape
    assert yhat.shape == y.shape

    for yhati in yhat:
        assert not np.isnan(yhati) and not np.abs(yhati) < 1e-15

    model.setTrainDataOnDevice(x,y)

    model.supervised_update(0,2)


def test_structured_linear_regression():

    schema = {'description':'test CG',
              'type': 'regression',
              'supervised-cost': {'type': 'absolute-error', 
                                  'name': 'vector-out'},
              'graph':
                  [{'activation':'linear',
                    'n_in':5,
                    'n_out':2,
                    'dropout':0.0,
                    'name':'vector-out'}]}

    model = ComputationalGraph(schema = schema,
                               seed = 0)

    nLayers = len(model.layers)

    assert nLayers == 1
    for i in xrange(nLayers):
        assert not np.any(model.layers[i].weights[1])
        
    assert model.targetType == theano.config.floatX
        
    x = np.asarray([[1,2,3,4,5],[2,3,4,5,6]]).astype(theano.config.floatX)
    y = np.asarray([[0,0.5],[1,0.2]],dtype=theano.config.floatX).astype(theano.config.floatX)
    
    yhat = model.predict(x)

    print yhat.shape
    assert yhat.shape == y.shape

    for yhati in yhat:
        assert not np.any(np.isnan(yhati)) and not np.any(np.abs(yhati)) < 1e-15

    model.setTrainDataOnDevice(x,y)

    model.supervised_update(0,2)
