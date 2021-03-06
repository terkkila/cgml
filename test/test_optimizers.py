
import theano
import theano.tensor as T

import cgml.types
from cgml.optimizers import Momentum,AdaDelta
from cgml.graph import ComputationalGraph
from cgml.layers.base import _make_shared

import numpy as np

from nose.tools import assert_true,assert_equals,assert_almost_equals

def test_momentum():

    schema = {'description':'logreg',
              'type':'classification',
              'supervised-cost':
                  {'type':'negative-log-likelihood',
                   'name':'class-out'},
              'graph':
                  [{'activation':'softmax',
                    'n_in':10,
                    'n_out':2,
                    'dropout':0.0,
                    'name':'class-out'}]
              }

    #optimizer = Momentum()

    #model = ComputationalGraph(schema = schema,
    #                           optimizer = optimizer)


def test_adadelta_logreg():

    x = T.fvector('x')
    y = T.fscalar('y')
    w = _make_shared([1.0,1.0],name='w')
    b = _make_shared([1.0],name='b')
    yhat = 1.0 / ( 1.0 + T.exp( - T.dot(x,w) - b ) )

    e = y - yhat

    cost = T.dot(e,e)

    ad = AdaDelta(cost = cost,
                  params = [w,b])

    update = theano.function( inputs  = [x,y],
                              outputs = cost,
                              updates = ad.updates )

    c = update([2,1],0)

    assert_almost_equals(c,0.9643510838246173)

    c_prev = c

    for i in range(100):
        c = update([2,1],0)
        assert_equals(c,c)
        assert_true(c < c_prev)
        c_prev = c
    

def test_adadelta_model():

    schema = {'description':'logreg',
              'type':'classification',
              'supervised-cost':
                  {'type':'negative-log-likelihood',
                   'name':'class-out'},
              'graph':
                  [{'activation':'softmax',
                    'n_in':10,
                    'n_out':2,
                    'dropout':0.0,
                    'name':'class-out'}]
              }
    
    model = ComputationalGraph(schema = schema,
                               seed = 0)
    
    x = np.asarray([[1,2,3,4,5,1,2,3,4,5]]).astype(cgml.types.floatX)
    y = np.asarray([0],dtype=cgml.types.intX)

    model.setTrainDataOnDevice(x,y)

    for i in range(10):
        model.supervised_update(0,1)

