
import numpy as np
import theano
import theano.tensor as T
from cgml.layers import Layer,ConvolutionLayer
from cgml.activations import activationMap
from cgml.costs import costMap,nllCost,sqerrCost,crossEntCost
from cgml.optimizers import MSGD
from cgml.schema import validateSchema

def parseLayers(x,schema,rng):

    schemaLayers = schema['graph']

    nLayers = len(schemaLayers)

    if schema.get('unsupervised-cost'):

        modelType = 'autoencoder'

        print "!!NOTE: temporarily assuming that graph having unsupervised cost is an autoencoder!!"

        if nLayers % 2 != 0:
            raise Exception("Autoencoder graph must have even number of layers")
        
        for i in xrange(nLayers/2):
            if schemaLayers[i]['n_in'] != schemaLayers[nLayers - 1 - i]['n_out']:
                raise Exception("Autoencoder graph must be symmetric")

    else:

        print "!!NOTE: temporarily assuming that graph having supervised cost is a classifier!!"

        modelType = 'classifier'


    # Layers of the graph
    dropoutLayers = []
    layers = []
    
    lastOutput = x
    lastNOut = schemaLayers[0]['n_in']

    for i in xrange(nLayers):

        currDropoutLayer = schema['graph'][i]

        if currDropoutLayer['activation'] != 'conv2d':
            
            dropoutLayers.append( Layer(rng        = rng,
                                        input      = lastOutput,
                                        n_in       = lastNOut,
                                        n_out      = currDropoutLayer['n_out'],
                                        activation = activationMap[
                        currDropoutLayer['activation'] ],
                                        randomInit = True,
                                        dropout    = currDropoutLayer['dropout']) )

        else:
            
            dropoutLayers.append( ConvolutionLayer(rng          = rng,
                                                   input        = lastOutput,
                                                   n_in         = lastNOut,
                                                   n_out        = currDropoutLayer['n_out'],
                                                   activation   = activationMap[
                        currDropoutLayer['activation'] ],
                                                   randomInit   = True,
                                                   dropout      = currDropoutLayer['dropout'],
                                                   n_filters    = currDropoutLayer['n_filters'],
                                                   filter_width = currDropoutLayer['filter_width'],
                                                   subsample    = currDropoutLayer['subsample']))
            
        if modelType == 'autoencoder' and i >= nLayers/2:
            dropoutLayers[-1].W = dropoutLayers[nLayers-1-i].W.T

        lastOutput = dropoutLayers[-1].output
        lastNOut   = dropoutLayers[-1].n_out

    lastOutput = x
    lastNOut = schemaLayers[0]['n_in']

    for i in xrange(nLayers):

        activationStr = schema['graph'][i]['activation']

        currDropoutLayer = dropoutLayers[i]
        
        q = 1 - currDropoutLayer.dropout
        
        if activationStr != 'conv2d':
            
            layers.append( Layer(rng = rng,
                                 input = lastOutput,
                                 n_in  = lastNOut,
                                 n_out = currDropoutLayer.n_out,
                                 activation = currDropoutLayer.activation,
                                 W = currDropoutLayer.W * q,
                                 b = currDropoutLayer.b,
                                 dropout = 0) )

        else:

            layers.append( ConvolutionLayer(rng = rng,
                                            input = lastOutput,
                                            n_in  = lastNOut,
                                            n_out = currDropoutLayer.n_out,
                                            activation = currDropoutLayer.activation,
                                            W = currDropoutLayer.W * q,
                                            dropout = 0,
                                            n_filters = currDropoutLayer.n_filters,
                                            filter_width = currDropoutLayer.filter_width,
                                            subsample = currDropoutLayer.subsample) )


        lastOutput = layers[-1].output
        lastNOut   = layers[-1].n_out
        
    return layers,dropoutLayers,modelType


def percent(x):
    return '{0:3.2f}'.format(100*x) + '%'
    
class ComputationalGraph(object):

    def __init__(self,
                 schema = None,
                 log   = None,
                 learnRate = None,
                 momentum = None,
                 L1Reg = 0.0,
                 L2Reg = 0.0,
                 seed = None):

        # Run schema validator before we do anything
        validateSchema(schema)

        # Input data is always a data matrix
        x = T.dmatrix('x')

        # Otherwise clamp the input to the input matrix
        self.input = x

        # Symbolic output vector
        # NOTE: Currently we assume classification
        y = T.lvector('y')

        self.seed = seed

        self.rng = np.random.RandomState(self.seed)

        # Schema to build the model from
        self.schema = schema

        if log:
            log.write('Loaded the following schema: ' +
                      str(self.schema) + '\n')

        # Parse layers from the schema. Input is needed to clamp
        # it with the first layer
        self.layers,self.dropoutLayers,modelType = parseLayers(self.input,
                                                               self.schema,
                                                               self.rng)

        # Get model type
        self.type = modelType

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

        self._setUpCostFunctions(x,y,L1Reg,L2Reg)

        self._setUpOptimizers(x,y,learnRate,momentum)

        self._setUpOutputs(x)

        self.optimizer = theano.opt.Optimizer()


    def _setUpCostFunctions(self,x,y,L1Reg,L2Reg):
        
        self.L1norm = T.sum([T.sum(abs(layer.W.flatten())) for layer in self.layers])
        self.L2norm = T.sum([T.sum(layer.W.flatten() ** 2) for layer in self.layers])
        
        self.reg = L1Reg * self.L1norm + L2Reg * self.L2norm

        self._unsupervised_cost = None
        self._supervised_cost = None
        
        if self.schema.get('supervised-cost'):
            cost = costMap[self.schema['supervised-cost']['type']]
            self._supervised_cost = cost(self.dropoutOutput,y) + self.reg

        if self.schema.get('unsupervised-cost'):
            cost = costMap[self.schema['unsupervised-cost']['type']]
            self._unsupervised_cost = cost(self.dropoutOutput,x) + self.reg
    
    
    def _setUpOptimizers(self,x,y,learnRate,momentum):

        if self._supervised_cost:
            
            self.supervised_optimizer = MSGD(
                cost      = self._supervised_cost,
                params    = self.params,
                learnRate = learnRate,
                momentum  = momentum)
            
            self.supervised_update = theano.function(
                inputs  = [x,y],
                outputs = self._supervised_cost,
                updates = self.supervised_optimizer.updates)
            
            self.supervised_cost = theano.function(
                inputs  = [x,y],
                outputs = self._supervised_cost)

        if self._unsupervised_cost:
            
            self.unsupervised_optimizer = MSGD(
                cost      = self._unsupervised_cost,
                params    = self.params,
                learnRate = learnRate,
                momentum  = momentum)
            
            self.unsupervised_update = theano.function(
                inputs  = [x],
                outputs = self._unsupervised_cost,
                updates = self.unsupervised_optimizer.updates)

            self.unsupervised_cost = theano.function(
                inputs  = [x],
                outputs = self._unsupervised_cost)
        

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

        if self.schema.get('supervised-cost'):
            supCostStr = self.schema['supervised-cost']['type']
        else:
            supCostStr = "None"

        if self.schema.get('unsupervised-cost'):
            unsupCostStr = self.schema['unsupervised-cost']['type']
        else:
            unsupCostStr = "None"

        return ("Computational graph:\n" +
                " - description : " + self.schema['description'] + '\n' +
                " - graph layout: " + graphStr                   + '\n' + 
                " - cost(sup.)  : " + supCostStr                 + '\n' + 
                " - cost(unsup.): " + unsupCostStr               + '\n' )





