
import theano.tensor as T
from layers import Layer

class LogRegClassifier(object):

    def __init__(self,x = None, n_in = None, n_out = None):
        """A Multi-class Logistic Regression Classifier.

        Input arguments:
         x               A symbolic variable denoting the input data
         n_in            How many features?
         n_out           How many classes?
        """

        # Logistic Regression Classifier only has one layer.
        # The transformation by the layer is softmax, which
        # gives us the class probabilities
        self.layer = Layer(input = x,
                           n_in  = n_in,
                           n_out = n_out,
                           activation = T.nnet.softmax)

        # Parameters equal to that of the logistic layer
        self.params = self.layer.params

        # Output of the layer are the class probabilities 
        self.y_prob = self.layer.output

        # Prediction is the class label with maximum probability
        self.y_pred = T.argmax(self.y_prob, axis = 1)
















