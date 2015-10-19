
import theano
import theano.tensor as T
import numpy as np
from nose.tools import assert_true

import cgml.types
from cgml.graph import ComputationalGraph
from cgml.makers import makeSchema
import matplotlib.pyplot as plt

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
        
    assert model.targetType == cgml.types.floatX
        
    x = np.asarray([[1,2,3,4,5],[2,3,4,5,6]]).astype(cgml.types.floatX)
    y = np.asarray([[0],[1]],dtype=cgml.types.floatX).astype(cgml.types.floatX)
    
    yhat = model.predict(x)

    print yhat.shape
    assert yhat.shape == y.shape

    for yhati in yhat:
        assert not np.isnan(yhati) and not np.abs(yhati) < 1e-15

    model.update(x, y)

    yhat2 = model.predict(x)


def test_linear_regression_accuracy():

    np.random.seed(0)
     
    X = np.array([[-1.0,-1.0], [1.0,1.0], [2.0,2.0], [3.0,3.0], [4.0,4.0], [5.0,5.0]])
    #Y = X[:,0].reshape((6,1))
    Y = np.array(X[:,0].reshape((6,1)), copy=True)
    
    X = np.tile(X, (10, 1))
    Y = np.tile(Y, (10, 1))
    
    X = X + 0.01 * np.random.randn(*X.shape)
    Y = Y + 0.01 * np.random.randn(*Y.shape)
    
    nFeatures = X.shape[1]
    nOutputs = Y.shape[1]

    miniBatchSize = 10
    nTimes = 1000
    
    for useDropout in [False]:

        schema = makeSchema(n_in=nFeatures,
                            n_out=nOutputs,
                            nLayers=2,
                            modelType="regression",
                            inputDropRate = 2,
                            costFunction = "squared-error",
                            activationFunction = "linear",
                            useDropout=useDropout)
        
        model = ComputationalGraph(schema=schema, seed=None)
        model.update(X, Y,
                     miniBatchSize=miniBatchSize,
                     nTimes=nTimes)
        
        YHat = model.predict(X)
        
        err = np.mean(np.abs(YHat.ravel() - Y.ravel()))
        
        assert_true(err < 0.04)
    


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
        
    assert model.targetType == cgml.types.floatX
        
    x = np.asarray([[1,2,3,4,5],[2,3,4,5,6]]).astype(cgml.types.floatX)
    y = np.asarray([[0,0.5],[1,0.2]],dtype=cgml.types.floatX).astype(cgml.types.floatX)
    
    yhat = model.predict(x)

    print yhat.shape
    assert yhat.shape == y.shape

    for yhati in yhat:
        assert not np.any(np.isnan(yhati)) and not np.any(np.abs(yhati)) < 1e-15

    model.setTrainDataOnDevice(x,y)

    model.supervised_update(0,2)
