
import numpy as np
import theano
import theano.tensor as T
from cgml.activations import linrect
from cgml.layers import Layer,DropoutLayer
from cgml.costs import nllCost,sqerrCost
from cgml.optimizers import MSGD
import yaml

allowedGraphs = ['classifier','regressor','autoencoder','reinforcement-learner']

activationMap = {'linear':  None,
                 'sigmoid': T.nnet.sigmoid,
                 'tanh':    T.tanh,
                 'softmax': T.nnet.softmax,
                 'linrect': linrect}

def parseLayers(x,schema):

    rng = np.random.RandomState(1234)

    assert schema['type'] in allowedGraphs

    schemaLayers = schema['graph']

    nLayers = len(schemaLayers)

    if schema['type'] == 'autoencoder':

        if nLayers % 2 != 0:
            raise Exception("Autoencoder graph must have even number of layers")
        
        for i in xrange(nLayers/2):
            if schemaLayers[i]['n_in'] != schemaLayers[nLayers - 1 - i]['n_out']:
                raise Exception("Autoencoder graph must be symmetric")

    # Layers of the graph
    dropoutLayers = []
    layers = []
    
    # Should we initialize the weights randomly, or set them to zero?
    randomInit = schema['randomInit']

    lastOutput = x
    lastNOut = schemaLayers[0]['n_in']

    for i in xrange(nLayers):

        currDropoutLayer = schema['graph'][i]
        
        dropoutLayers.append( DropoutLayer(rng        = rng,
                                           input      = lastOutput,
                                           n_in       = lastNOut,
                                           n_out      = currDropoutLayer['n_out'],
                                           activation = activationMap[
                                               currDropoutLayer['activation'] ],
                                           randomInit = randomInit,
                                           p          = currDropoutLayer['dropout']) )

        if schema['type'] == 'autoencoder' and i >= nLayers/2:
            dropoutLayers[-1].W = dropoutLayers[nLayers-1-i].W.T

        lastOutput = dropoutLayers[-1].output
        lastNOut   = dropoutLayers[-1].n_out


    lastOutput = x
    lastNOut = schemaLayers[0]['n_in']

    for i in xrange(nLayers):

        currDropoutLayer = dropoutLayers[i]
        
        q = 1 - currDropoutLayer.p
        
        layers.append( Layer(rng = rng,
                             input = lastOutput,
                             n_in  = lastNOut,
                             n_out = currDropoutLayer.n_out,
                             activation = currDropoutLayer.activation,
                             W = currDropoutLayer.W * q,
                             b = currDropoutLayer.b))

        lastOutput = layers[-1].output
        lastNOut   = layers[-1].n_out
        
    return layers,dropoutLayers


def percent(x):
    return '{0:3.2f}'.format(100*x) + '%'
    
class ComputationalGraph(object):

    def __init__(self,
                 x = None,
                 y = None,
                 cg    = None,
                 log   = None,
                 learnRate = None):

        # Take symbolic representation of the input data
        self.input = x

        # Load schema from input file
        self.schema = yaml.load(open(cg,'r'))

        if log:
            log.write('Loaded the following schema: ' +
                      str(self.schema) + '\n')

        # Get model type
        self.type = self.schema['type']

        # Parse layers from the schema. Input is needed to clamp
        # it with the first layer
        self.layers,self.dropoutLayers = parseLayers(self.input,self.schema)

        # Take number of input variables
        self.n_in = self.layers[0].n_in
        
        # Collect parameters of all the layers
        self.params = [param for layer in self.dropoutLayers
                       for param in layer.params]
        
        # Clamp output to the output of the last layer
        self.output = self.layers[-1].output

        # Clamp dropout output to the output of the last dropout layer
        self.dropoutOutput = self.dropoutLayers[-1].output
        
        # The number of outputs is obtained from the output layer of the graph
        self.n_out  = self.schema['graph'][-1]['n_out']

        self._setUpCostFunctions(x,y)

        self._setUpOptimizers(x,y,learnRate)

        self._setUpOutputs(x)


    def _setUpCostFunctions(self,x,y):

        self.unsupervised_cost = None
        self.supervised_cost = None

        if self.type == 'classifier':
            self.supervised_cost = nllCost(self.dropoutOutput,y)
        elif self.type == 'regressor':
            self.supervised_cost = sqerrCost(self.dropoutOutput,y)

        self.unsupervised_cost = sqerrCost(self.dropoutOutput,x)

    
    def _setUpOptimizers(self,x,y,learnRate):

        if self.supervised_cost:
            
            self.supervised_optimizer = MSGD(
                cost      = self.supervised_cost,
                params    = self.params,
                learnRate = learnRate)
            
            self.supervised_update = theano.function(
                inputs  = [x,y],
                outputs = self.supervised_cost,
                updates = self.supervised_optimizer.updates)
            
        if self.unsupervised_cost:
            
            self.unsupervised_optimizer = MSGD(
                cost      = self.unsupervised_cost,
                params    = self.params,
                learnRate = learnRate)
            
            self.unsupervised_update = theano.function(
                inputs  = [x],
                outputs = self.unsupervised_cost,
                updates = self.unsupervised_optimizer.updates)


    def _setUpOutputs(self,x):
        
        self.encode = None
        self.predict = None

        if self.type == 'classifier':
        
            self.predict = theano.function( inputs = [x],
                                            outputs = T.argmax(self.output,
                                                               axis = 1) )

        elif self.type == 'regressor':
                    
            self.predict = theano.function( inputs = [x],
                                            outputs = self.output )

        elif self.type == 'autoencoder':

            self.predict = theano.function( inputs = [x],
                                            outputs = self.output )

            
            self.encode = theano.function( inputs = [x],
                                           outputs = self.layers[self.encodeLayerIdx()].output )


    def encodeLayerIdx(self):
        assert self.type == 'autoencoder'
        return len(self.layers) / 2 - 1


    def __str__(self):

        graphList = ['input(' + str(self.schema['graph'][0]['n_in']) + ')']
        
        for layer in self.schema['graph']:
            
            graphList.append('drop ' + percent(layer['dropout']) +
                             ' -> ' + layer['activation'] +
                              '(' + str(layer['n_out']) + ')')

        graphStr = ' '.join(graphList)

        if self.type == 'classifier':
            supCostStr = 'Negative log-likelihood'
        elif self.type == 'regressor':
            supCostStr = 'Squared error'
        else:
            supCostStr = 'None'

        unsupCostStr = 'Squared error'
            
        return ("Computational graph:\n" +
                " - description : " + self.schema['description'] + '\n' +
                " - type        : " + self.schema['type']        + '\n' +
                " - graph layout: " + graphStr                   + '\n' + 
                " - cost(sup.)  : " + supCostStr                 + '\n' + 
                " - cost(unsup.): " + unsupCostStr               + '\n' )








