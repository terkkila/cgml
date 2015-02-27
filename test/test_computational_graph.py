
import numpy as np
from cgml.graph import ComputationalGraph

from nose.tools import assert_equals,assert_true,assert_almost_equals

def test_cg():

    schema = {'description':'test CG',
              'type':'classification',
              'supervised-cost': {'type': 'negative-log-likelihood',
                                  'name': 'class-out'},
              'graph':
                  [{'activation':'linear',
                    'n_in':10,
                    'n_out':5,
                    'dropout':0.0,
                    'name':'hidden1'},
                   {'activation':'sigmoid',
                    'n_in':5,
                    'n_out':2,
                    'dropout':0.0,
                    'name':'class-out'}]}
     
    model = ComputationalGraph(schema = schema,
                               seed = 0)

    

def test_cg_serde():

    schema = {'description':'test CG',
              'type':'autoencoder',
              'supervised-cost': {'type': 'cross-entropy',
                                  'name': 'decode-out'},
              'graph':
                  [{'activation':'linear',
                    'n_in':2,
                    'n_out':1,
                    'dropout':0.0,
                    'name':'encode-out'},
                   {'activation':'sigmoid',
                    'n_in':1,
                    'n_out':2,
                    'dropout':0.0,
                    'name':'decode-out'}]}
    
    model = ComputationalGraph(schema = schema)

    schema2 = model.getSchema()

    assert_equals(len(schema2['graph']),2)
    assert_equals(schema2['graph'][0]['W'].shape,(2,1))
    assert_equals(schema2['graph'][0]['b'].shape,(1,))
    assert_equals(schema2['graph'][1]['W'].shape,(1,2))
    assert_equals(schema2['graph'][1]['b'].shape,(2,))


