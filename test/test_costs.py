
import theano
import theano.tensor as T
import numpy as np
from cgml.costs import sqerrCost,absCost,nllCost

def test_squared_error_cost():

    ySym,yhatSym = T.fmatrices('y','yhat')

    sqerr = theano.function([yhatSym,ySym],
                            outputs=sqerrCost(yhatSym,ySym))

    yhat = np.asarray([[1],[2],[3]],dtype=theano.config.floatX)
    y = np.asarray([[1],[2],[3]],dtype=theano.config.floatX)

    assert np.abs(sqerr(yhat,y)) < 1e-5

    yhat = np.asarray([[1],[2.1],[3]],dtype=theano.config.floatX)

    assert np.abs(sqerr(yhat,y) - 0.01/3) < 1e-5

    
def test_abs_cost():

    ySym,yhatSym = T.fmatrices('y','yhat')

    ac = theano.function([yhatSym,ySym],
                         outputs=absCost(yhatSym,ySym))
    
    yhat = np.asarray([[1],[2],[3]],dtype=theano.config.floatX)
    y = np.asarray([[1],[2],[3]],dtype=theano.config.floatX)

    assert np.abs(ac(yhat,y)) < 1e-5

    yhat = np.asarray([[1],[2.1],[3]],dtype=theano.config.floatX)

    assert np.abs(ac(yhat,y) - 0.1/3) < 1e-5

    
def test_nll_cost():

    yhatSym = T.fmatrix('yhat')
    ySym = T.lvector('y')

    nllc = theano.function([yhatSym,ySym],
                           outputs=nllCost(yhatSym,ySym))
    
    yhat = np.asarray([[0,1,0],[0,0,1]],dtype=theano.config.floatX)
    y = np.asarray([1,2],dtype=np.int)

    assert np.abs(nllc(yhat,y)) < 1e-5

    yhat = np.asarray([[0.1,0.8,0.1],[0.1,0.2,0.7]],dtype=theano.config.floatX)

    assert np.abs(nllc(yhat,y) - 0.2899) < 1e-5

    
