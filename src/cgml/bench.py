
import theano
import theano.tensor as T
from data import shareData

def trainTestBench(x = None,
                   y = None,
                   x_train = None,
                   y_train = None,
                   x_test = None,
                   y_test = None,
                   model = None,
                   cost = None,
                   optimizer = None,
                   error = None,
                   verbose = False):

    x_train_sh,y_train_sh = shareData( (x_train,y_train) )
    x_test_sh,y_test_sh   = shareData( (x_test,y_test) )

    n_train = x_train.shape[0]
    n_test  = x_test.shape[0]
    
    predict = theano.function(inputs = [x], outputs = model.y_pred)
    
    idx1,idx2 = T.iscalars('idx1','idx2')
    
    train_model = theano.function(inputs  = [idx1,idx2],
                                  outputs = cost,
                                  updates = optimizer.updates,
                                  givens  = {x:x_train_sh[idx1:idx2],y:y_train_sh[idx1:idx2]} )
    
    
    test_model = theano.function(inputs  = [idx1,idx2],
                                 outputs = error,
                                 givens  = {x:x_test_sh[idx1:idx2],y:y_test_sh[idx1:idx2]} )
    
    nEpochs = 50

    if verbose:
        print "\nBEFORE TRAINING"
        print "Prediction:  ", predict(x_train)
        print "Ground truth:", y_train
        print "Error:       ", test_model(0,n_test)
    
    for epoch in range(nEpochs):
        for idx in range(n_train-5):
            train_model(idx,idx+5)

    if verbose:
        print "\nAFTER TRAINING"
        print "Prediction:  ", predict(x_train)
        print "Ground truth:", y_train
        print "Error:       ", test_model(0,n_test)
        

