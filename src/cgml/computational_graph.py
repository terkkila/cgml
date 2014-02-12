
import numpy as np
import theano.tensor as T
from cgml.activations import linrect
from cgml.layers import Layer,DropoutLayer
import yaml

allowedGraphs = ['classifier','regressor','autoencoder','reinforcement-learner']

activationMap = {'linear':  None,
                 'sigmoid': T.nnet.sigmoid,
                 'tanh':    T.tanh,
                 'softmax': T.nnet.softmax,
                 'linrect': linrect}
    
def parseLayers(x,schema):

    rng = np.random.RandomState(1234)

    # Layers of the graph
    dropoutLayers = []
    layers = []
    
    # Should we initialize the weights randomly, or set them to zero?
    randomInit = schema['randomInit']

    # Obtain a shortcut to the first layer
    currDropoutLayer = schema['graph'][0]
    
    # Create the first layer of the graph
    dropoutLayers.append(
        DropoutLayer(rng        = rng,
                     input      = x,
                     n_in       = currDropoutLayer['n_in'],
                     n_out      = currDropoutLayer['n_out'],
                     activation = activationMap[
                         currDropoutLayer['activation'] ],
                     randomInit = randomInit,
                     p          = currDropoutLayer['dropout']) )

    # Create subsequent layers of the graph
    for i in xrange(1,len(schema['graph'])):

        prevDropoutLayer = dropoutLayers[i-1]
        currDropoutLayer = schema['graph'][i]
        
        dropoutLayers.append( DropoutLayer(rng        = rng,
                                           input      = prevDropoutLayer.output,
                                           n_in       = prevDropoutLayer.n_out,
                                           n_out      = currDropoutLayer['n_out'],
                                           activation = activationMap[
                                               currDropoutLayer['activation'] ],
                                           randomInit = randomInit,
                                           p          = currDropoutLayer['dropout']) )

    q = 1 - dropoutLayers[0].p
        
    layers = [ Layer(rng = rng,
                     input = x,
                     n_in  = dropoutLayers[0].n_in,
                     n_out = dropoutLayers[0].n_out,
                     activation = dropoutLayers[0].activation,
                     W = dropoutLayers[0].W * q,
                     b = dropoutLayers[0].b) ]

    prevLayer = layers[0]
    
    for i in xrange(1,len(dropoutLayers)):

        currDropoutLayer = dropoutLayers[i]
        
        q = 1 - currDropoutLayer.p
        
        layers.append( Layer(rng = rng,
                             input = prevLayer.output,
                             n_in  = prevLayer.n_out,
                             n_out = currDropoutLayer.n_out,
                             activation = currDropoutLayer.activation,
                             W = currDropoutLayer.W * q,
                             b = currDropoutLayer.b))

        prevLayer = layers[-1]
        
    return layers,dropoutLayers


def percent(x):
    return '{0:3.2f}'.format(100*x) + '%'
    
class ComputationalGraph(object):

    def __init__(self,
                 input = None,
                 cg    = None,
                 log   = None):

        # Take symbolic representation of the input data
        self.input = input

        # Load schema from input file
        self.schema = yaml.load(open(cg,'r'))

        if log:
            log.write('Loaded the following schema: ' +
                      str(self.schema) + '\n')

        # Parse layers from the schema. Input is needed to clamp
        # it with the first layer
        self.layers,self.dropoutLayers = parseLayers(self.input,self.schema)

        # Take number of input variables
        self.n_in = self.layers[0].n_in
        
        # Collect parameters of all the layers
        self.params = [param for layer in self.dropoutLayers
                       for param in layer.params]
        
        # Clamp output to the output of the last layer of the graph
        self.output = self.layers[-1].output

        self.dropoutOutput = self.dropoutLayers[-1].output
        
        # The number of outputs is obtained from the output layer of the graph
        self.n_out  = self.schema['graph'][-1]['n_out']

    def __str__(self):

        graphList = ['input(' + str(self.schema['graph'][0]['n_in']) + ')']
        
        for layer in self.schema['graph']:
            
            graphList.append('drop ' + percent(layer['dropout']) +
                             ' -> ' + layer['activation'] +
                              '(' + str(layer['n_out']) + ')')

        graphStr = ' '.join(graphList)
            
        return ("Computational graph:\n" +
                " - description : " + self.schema['description'] + '\n' +
                " - type        : " + self.schema['type']        + '\n' +
                " - graph layout: " + graphStr )








