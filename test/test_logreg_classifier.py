

import theano.tensor as T

from data import makeRandomClassificationData
from optimizers import MSGD
from classifiers import LogRegClassifier
from costs import negativeLogLikelihood
from errors import misclassificationRate
from bench import trainTestBench

n = 20
n_in  = 5
n_out = 3
learnRate = 0.1

x = T.dmatrix('x')
y = T.ivector('y')

# Define a Logistic Regression classifier with n_in input variables, and n_out output variables
# x denotes the symbolic representation of the input data, which is a matrix
logreg_classifier = LogRegClassifier(x     = x,
                                     n_in  = n_in,
                                     n_out = n_out)

# Define cost as the negative log-likelihood
nll_cost = negativeLogLikelihood(y_prob = logreg_classifier.y_prob,
                                 y      = y)

# Use mini-batch stochastic gradient descent to optimize parameters 
msgd_optimizer = MSGD(cost      = nll_cost,
                      params    = logreg_classifier.params,
                      learnRate = learnRate)

# Define how to quantify error
misclass_error = misclassificationRate(y_pred = logreg_classifier.y_pred,
                                       y      = y)

# Some random data 
x_train,y_train = makeRandomClassificationData(n     = n,
                                               n_in  = n_in,
                                               n_out = n_out)

# Start the test bench
trainTestBench(x         = x,
               y         = y,
               x_train   = x_train,
               y_train   = y_train,
               x_test    = x_train,
               y_test    = y_train,
               model     = logreg_classifier,
               cost      = nll_cost,
               optimizer = msgd_optimizer,
               error     = misclass_error,
               verbose   = True)














