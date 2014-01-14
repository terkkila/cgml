
import theano.tensor as T

def misclassificationRate(y_pred,y):
    """Return a float representing the number of errors in the minibatch
    over the total number of examples of the minibatch ; zero
    one loss over the size of the minibatch
    """
    
    return T.mean( T.neq(y_pred, y) )
    










