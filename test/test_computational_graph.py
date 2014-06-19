
import numpy as np
from cgml.computational_graph import ComputationalGraph

from nose.tools import assert_equals,assert_true,assert_almost_equals

def test_computational_graph():

    schema = {'description':'test CG',
              'supervised-cost': {'type': 'negative-log-likelihood',
                                  'name': 'class-out'},
              'graph':
                  [{'activation':'linear',
                    'n_in':10,
                    'n_out':5,
                    'dropout':0.0},
                   {'activation':'sigmoid',
                    'n_in':5,
                    'n_out':2,
                    'dropout':0.0,
                    'name':'class-out'}]}
     
    model = ComputationalGraph(schema = schema,
                               seed = 0)

    rng = np.random.RandomState(0)

    #X = rng.

    #model.setTrainDataOnDevice()
