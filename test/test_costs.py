
import theano
import theano.tensor as T
import numpy as np
from cgml.costs import sqerrCost,absCost

def test_squared_error_cost():

    ySym,yhatSym = T.fvectors('y','yhat')

    sqerr = theano.function([yhatSym,ySym],
                            outputs=sqerrCost(yhatSym,ySym))

    yhat = np.asarray([1,2,3],dtype=theano.config.floatX)
    y = np.asarray([1,2,3],dtype=theano.config.floatX)

    assert np.abs(sqerr(yhat,y)) < 1e-5

    yhat = np.asarray([1,2.1,3],dtype=theano.config.floatX)

    assert np.abs(sqerr(yhat,y) - 0.01/3) < 1e-5

    
def test_abs_cost():

    ySym,yhatSym = T.fvectors('y','yhat')

    ac = theano.function([yhatSym,ySym],
                         outputs=absCost(yhatSym,ySym))
    
    yhat = np.asarray([1,2,3],dtype=theano.config.floatX)
    y = np.asarray([1,2,3],dtype=theano.config.floatX)

    assert np.abs(ac(yhat,y)) < 1e-5

    yhat = np.asarray([1,2.1,3],dtype=theano.config.floatX)

    assert np.abs(ac(yhat,y) - 0.1/3) < 1e-5

    
