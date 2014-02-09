
import numpy as np
import theano.tensor as T
from layers import Layer
import yaml

allowedGraphs = ['classifier','regressor','autoencoder','reinforcement-learner']

activationMap = {'linear':  None,
                 'sigmoid': T.nnet.sigmoid,
                 'tanh':    T.tanh,
                 'softmax': T.nnet.softmax}

def parseLayerStr(layerStr):

    try:
        elems      = layerStr.split(' ')
        activation = activationMap[elems[0]]
        n_out      = int(elems[1])
    except:
        raise Exception('Cannot parse layer in graph: ' + layerStr)

    return activation,n_out

    
def parseLayers(x,schema):

    rng = np.random.RandomState(1234)
    
    # Layers of the graph
    layers = []

    # Number of inputs 
    n_in = schema['n_in']

    # Should we initialize the weights randomly, or set them to zero?
    randomInit = schema['randomInit']
    
    # Parse the type of the activation function, and dimensionality
    # of the output of the activation (maps to input of next layer)
    activation,n_out = parseLayerStr(schema['graph'][0])

    # Create the first layer of the graph
    layers.append( Layer(rng        = rng,
                         input      = x,
                         n_in       = n_in,
                         n_out      = n_out,
                         activation = activation,
                         randomInit = randomInit) )

    # Create subsequent layers of the graph
    for i in xrange(1,len(schema['graph'])):

        n_in = layers[i-1].n_out

        activation,n_out = parseLayerStr(schema['graph'][i])

        layers.append( Layer(rng        = rng,
                             input      = layers[i-1].output,
                             n_in       = n_in,
                             n_out      = n_out,
                             activation = activation,
                             randomInit = randomInit) )

    if layers[-1].n_out != schema['n_out']:
        raise Exception('Dimensionality of the output of the last layer (' +
                        str(layers[-1].n_out) + ') does not match that ' +
                        'specified in the schema (' + str(schema['n_out']) +
                        ')')

    return layers
        
    
class ComputationalGraph(object):

    def __init__(self,x,fileName):

        # Take symbolic representation of the input data
        self.input = x

        # Load schema from input file
        self.schema = yaml.load(open(fileName,'r'))

        # Take the number of inputs and outputs from the schema
        self.n_in  = self.schema['n_in']
        self.n_out = self.schema['n_out']

        # Parse layers from the schema. Input is needed to clamp
        # it with the first layer
        self.layers = parseLayers(self.input,self.schema)

        # Collect parameters of all the layers
        self.params = []
        for layer in self.layers:
            self.params += layer.params

        # Clamp output to the output of the last layer of the graph
        self.output = self.layers[-1].output

    def __str__(self):

        return ("Computational graph:\n" +
                " - description : " + self.schema['description'] + '\n' +
                " - type        : " + self.schema['type']        + '\n' +
                " - graph layout: " + str(self.schema['graph'])  + '\n' )




















