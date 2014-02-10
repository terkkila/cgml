
import numpy as np
import theano.tensor as T
from layers import Layer
import yaml

allowedGraphs = ['classifier','regressor','autoencoder','reinforcement-learner']

activationMap = {'linear':  None,
                 'sigmoid': T.nnet.sigmoid,
                 'tanh':    T.tanh,
                 'softmax': T.nnet.softmax}
    
def parseLayers(x,schema):

    rng = np.random.RandomState(1234)

    # Layers of the graph
    layers = []

    # Should we initialize the weights randomly, or set them to zero?
    randomInit = schema['randomInit']

    # Obtain a shortcut to the first layer
    currLayer = schema['graph'][0]
    
    # Create the first layer of the graph
    layers.append( Layer(rng        = rng,
                         input      = x,
                         n_in       = currLayer['n_in'],
                         n_out      = currLayer['n_out'],
                         activation = activationMap[ currLayer['activation'] ],
                         randomInit = randomInit) )

    # Create subsequent layers of the graph
    for i in xrange(1,len(schema['graph'])):

        prevLayer = layers[i-1]
        currLayer = schema['graph'][i]
        
        layers.append( Layer(rng        = rng,
                             input      = prevLayer.output,
                             n_in       = prevLayer.n_out,
                             n_out      = currLayer['n_out'],
                             activation = activationMap[currLayer['activation']],
                             randomInit = randomInit) )

    return layers
        
    
class ComputationalGraph(object):

    def __init__(self,x,fileName):

        # Take symbolic representation of the input data
        self.input = x

        # Load schema from input file
        self.schema = yaml.load(open(fileName,'r'))

        # Take the number of inputs from the schema
        self.n_in  = self.schema['input']['n_in']

        # Parse layers from the schema. Input is needed to clamp
        # it with the first layer
        self.layers = parseLayers(self.input,self.schema)

        # Collect parameters of all the layers
        self.params = []
        for layer in self.layers:
            self.params += layer.params

        # Clamp output to the output of the last layer of the graph
        self.output = self.layers[-1].output

        # The number of outputs is obtained from the output layer of the graph
        self.n_out  = self.schema['graph'][-1]['n_out']

    def __str__(self):

        return ("Computational graph:\n" +
                " - description : " + self.schema['description'] + '\n' +
                " - type        : " + self.schema['type']        + '\n' +
                " - graph layout: " + str(self.schema['graph'])  + '\n' )




















