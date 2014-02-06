
import numpy.random
import theano
import theano.tensor as T
from data import makeShared
from classifiers import LogRegClassifier, MultiLayerPerceptronClassifier

def trainTestBench(x = None,
                   y = None,
                   x_train = None,
                   y_train = None,
                   x_test = None,
                   y_test = None,
                   model = None,
                   cost = None,
                   optimizer = None,
                   verbose = False):

    x_train_sh = makeShared( x_train )
    y_train_sh = makeShared( y_train )
    
    x_test_sh  = makeShared( x_test )
    y_test_sh  = makeShared( y_test )
    
    if type(model) == LogRegClassifier or type(model) == MultiLayerPerceptronClassifier:
        y_train_sh = T.cast(y_train_sh, 'int32')
        y_test_sh  = T.cast(y_test_sh, 'int32')
        predict = theano.function(inputs = [x], outputs = T.argmax(model.output, axis=1) )
    else:
        predict = theano.function(inputs = [x], outputs = model.output)

    
    n_train = x_train.shape[0]
    n_test  = x_test.shape[0]
    
    ics = T.lvector('ics')
    
    train_model = theano.function(inputs  = [ics],
                                  outputs = cost,
                                  updates = optimizer.updates,
                                  givens  = {x:x_train_sh[ics],y:y_train_sh[ics]} )
    
    
    test_model = theano.function(inputs  = [ics],
                                 outputs = cost,
                                 givens  = {x:x_test_sh[ics],y:y_test_sh[ics]} )

    nDraws = n_train
    nMiniBatch = 5

    if verbose:
        print "\nBEFORE TRAINING"
        print "Prediction:  ", predict(x_train)
        print "Ground truth:", y_train
        print "Cost:       ",  test_model(range(n_test))
    
    for draw in range(nDraws):
        ics = numpy.random.randint(0,n_train,nMiniBatch)
        train_model(ics)

    if verbose:
        print "\nAFTER TRAINING"
        print "Prediction:  ", predict(x_test)
        print "Ground truth:", y_test
        print "Cost:       ",  test_model(range(n_test))
        











