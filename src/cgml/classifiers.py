
import theano.tensor as T
from layers import Layer

class LogRegClassifier(object):

    def __init__(self,
                 x = None,
                 n_in = None,
                 n_out = None,
                 randomInit = True):
        """A Multi-class Logistic Regression Classifier.

        Input arguments:
         x               A symbolic variable denoting the input data
         n_in            How many features?
         n_out           How many classes?
         randomInit      Do we want to initialize the weights randomly
        """

        # Logistic Regression Classifier only has one layer.
        # The transformation by the layer is softmax, which
        # gives us the class probabilities
        self.layer = Layer(input = x,
                           n_in  = n_in,
                           n_out = n_out,
                           activation = T.nnet.softmax,
                           randomInit = randomInit)

        # Parameters equal to that of the logistic layer
        self.params = self.layer.params

        # Output of the layer are the class probabilities 
        self.output = self.layer.output

class MultiLayerPerceptronClassifier(object):

    def __init__(self,
                 x = None,
                 n_in = None,
                 n_out = None,
                 randomInit = True):

        n_out1 = min(n_out,n_in/2)
        
        self.layer1 = Layer(input = x,
                            n_in = n_in,
                            n_out = n_out1,
                            activation = T.tanh,
                            randomInit = randomInit)

        self.layer2 = Layer(input = self.layer1.output,
                            n_in = n_out1,
                            n_out = n_out,
                            activation = T.nnet.softmax,
                            randomInit = randomInit)
        
        self.params = self.layer1.params + self.layer2.params

        self.output = self.layer2.output
