
import numpy as np
import theano
import theano.tensor as T
from cgml.layers import Layer,ConvolutionLayer
from cgml.activations import activationMap
from cgml.costs import costMap
from cgml.optimizers import MSGD
from cgml.schema import validateSchema

class DAG(object):


    def __init__(self):

        self.nodes = []
        self.childrenByIdx = {}
        self.name2idx = {}


    def addNode(self,node):

        if not self.name2idx.get(node.name):
            idx = len(self.nodes)
            self.name2idx[name] = idx
            self.nodes.append(node)
            self.childrenByIdx[idx] = []


    def addEdge(self,parent,child):

        self.addNode(parent)
        self.addNode(child)

        parentIdx = self.name2idx[parent.name]
        childIdx  = self.name2idx[child.name]

        self.childrenByIdx[parentIdx].append(childIdx)

        


    def getNode(self,name):
        return self.nodes[ self.name2idx[name] ]

        
        

def parseLayers(x,schema,rng):

    schemaLayers = schema['graph']

    nLayers = len(schemaLayers)

    if schema.get('unsupervised-cost') and schema.get('supervised-cost'):
        
        print "!!NOTE: temporarily assuming that graph having hybrid cost is a supervised autoencoder!!"
        
        modelType = 'supervised-autoencoder'

    elif schema.get('unsupervised-cost'):

        modelType = 'autoencoder'

        print "!!NOTE: temporarily assuming that graph having unsupervised cost is an autoencoder!!"

        if nLayers % 2 != 0:
            raise Exception("Autoencoder graph must have even number of layers")
        
        for i in xrange(nLayers/2):
            if schemaLayers[i]['n_in'] != schemaLayers[nLayers - 1 - i]['n_out']:
                raise Exception("Autoencoder graph must be symmetric")

    elif schema.get('supervised-cost'):

        print "!!NOTE: temporarily assuming that graph having supervised cost is a classifier!!"

        modelType = 'classifier'


    # Layers of the graph
    dropoutLayers = []
    layers = []

    branchDropoutLayer = None
    branchLayer = None
    
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
                                        dropout    = currDropoutLayer['dropout'],
                                        name       = currDropoutLayer.get('name',None)) )

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
                                                   subsample    = currDropoutLayer['subsample'],
                                                   name         = currDropoutLayer.get('name',None)))
            
        if modelType in ['autoencoder','supervised-autoencoder'] and i >= nLayers/2:
            dropoutLayers[-1].W = dropoutLayers[nLayers-1-i].W.T

        if currDropoutLayer.get('branch'):

            assert currDropoutLayer['branch'][0]['activation'] != 'conv2d'

            branchDropoutLayer = Layer(rng        = rng,
                                       input      = lastOutput,
                                       n_in       = lastNOut,
                                       n_out      = currDropoutLayer['branch'][0]['n_out'],
                                       activation = activationMap[
                    currDropoutLayer['branch'][0]['activation'] ],
                                       randomInit = True,
                                       dropout    = currDropoutLayer['branch'][0]['dropout'],
                                       name       = currDropoutLayer['branch'][0].get('name',None))
            
        lastOutput = dropoutLayers[-1].output
        lastNOut   = dropoutLayers[-1].n_out

    lastOutput = x
    lastNOut = schemaLayers[0]['n_in']

    graphHasBranch = False
    layerHasBranch = False

    for i in xrange(nLayers):

        layerHasBranch = (True if schema['graph'][i].get('branch') else False)

        if layerHasBranch:
            graphHasBranch = True

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
                                 dropout = 0,
                                 name = currDropoutLayer.name) )

            if layerHasBranch:
                branchLayer = Layer(rng   = rng,
                                    input = lastOutput,
                                    n_in  = lastNOut,
                                    n_out = branchDropoutLayer.n_out,
                                    activation = branchDropoutLayer.activation,
                                    W = branchDropoutLayer.W * q,
                                    b = branchDropoutLayer.b,
                                    dropout = 0,
                                    name = branchDropoutLayer.name)

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
                                            subsample = currDropoutLayer.subsample,
                                            name = currDropoutLayer.name) )


        lastOutput = layers[-1].output
        lastNOut   = layers[-1].n_out
        
    if graphHasBranch:
        layers.append(branchLayer)
        dropoutLayers.append(branchDropoutLayer)

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
        self.layers,self.dropoutLayers,modelType = parseLayers(self.input,self.schema,self.rng)

        # Get model type
        self.type = modelType

        # Collect parameters of all the layers
        self.params = [param for layer in self.dropoutLayers
                       for param in layer.params]
                
        self._setUpOutputs(x)

        self._setUpCostFunctions(x,y,L1Reg,L2Reg)

        self._setUpOptimizers(x,y,learnRate,momentum)

        self.optimizer = theano.opt.Optimizer()


    def _setUpOutputs(self,x):

        # We will use these for supervised and unsupervised learning
        self._supervised_dropout_output = None
        self._unsupervised_dropout_output = None

        # We will use these for supervised and unsupervised learning
        self._supervised_output = None
        self._unsupervised_output = None        

        # We will use these for getting the output
        self._encode_output = None
        self._decode_output = None

        # We will make these callable functions
        self.encode = None
        self.decode = None
        self.predict = None

        for layer,dropoutLayer in zip(self.layers,self.dropoutLayers):

            # If we find a layer that has supervised cost associated with it,
            # we will bind the respective output variables to that layer
            if ( self.schema.get('supervised-cost') and 
                 self.schema['supervised-cost']['name'] == dropoutLayer.name ):
                self._supervised_dropout_output = dropoutLayer.output
                self._supervised_output = layer.output
                self.predict = theano.function( inputs = [x],
                                                outputs = T.argmax(self._supervised_output,
                                                                   axis = 1) )


            # If we find a layer that as unsupervised cost associated with it,
            # we will bind the respective output variables to that layer
            if ( self.schema.get('unsupervised-cost') and 
                 self.schema['unsupervised-cost']['name'] == dropoutLayer.name ):
                self._unsupervised_dropout_output = dropoutLayer.output
                self._unsupervised_output = layer.output
        
            # In case encoder layer has been specified, we will bind 
            # the output variables to that
            if layer.name == 'encode-out':
                self._encode_output = layer.output
                self.encode = theano.function( inputs = [x],
                                               outputs = self._encode_output )


            # In case decoder layer has been specified, we will bind
            # the output variables to that
            if layer.name == 'decode-out':
                self._decode_output = layer.output
                self.decode = theano.function( inputs = [x],
                                               outputs = self._decode_output )

                
    def _setUpCostFunctions(self,x,y,L1Reg,L2Reg):
        
        self.L1norm = T.sum([T.sum(abs(layer.W.flatten())) for layer in self.layers])
        self.L2norm = T.sum([T.sum(layer.W.flatten() ** 2) for layer in self.layers])
        
        self.reg = L1Reg * self.L1norm + L2Reg * self.L2norm

        self._unsupervised_cost = None
        self._supervised_cost = None
        self._hybrid_cost = None
        
        if self._supervised_dropout_output:
            cost = costMap[self.schema['supervised-cost']['type']]
            self._supervised_cost = cost(self._supervised_dropout_output,y) + self.reg
            self.supervised_cost = theano.function(inputs = [x,y],
                                                   outputs = self._supervised_cost)

        if self._unsupervised_dropout_output:
            cost = costMap[self.schema['unsupervised-cost']['type']]
            self._unsupervised_cost = cost(self._unsupervised_dropout_output,x) + self.reg
            self.unsupervised_cost = theano.function(inputs = [x],
                                                     outputs = self._unsupervised_cost)

    
        if self._supervised_cost and self._unsupervised_cost:
            self._hybrid_cost = self._supervised_cost + self._unsupervised_cost
            self.hybrid_cost = theano.function(inputs = [x,y],
                                               outputs = self._hybrid_cost)

    
    def _setUpOptimizers(self,x,y,learnRate,momentum):

        self.supervised_update = None

        self.unsupervised_update = None

        self.hybrid_update = None

        if self._hybrid_cost:

            self.hybrid_optimizer = MSGD(
                cost      = self._hybrid_cost,
                params    = self.params,
                learnRate = learnRate,
                momentum  = momentum)

            self.hybrid_update = theano.function(
                inputs  = [x,y],
                outputs = self._hybrid_cost,
                updates = self.hybrid_optimizer.updates)

            return

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
        

    def __str__(self):

        graphList = ['input(' + str(self.schema['graph'][0]['n_in']) + ')']
        
        for layer in self.schema['graph']:
            
            graphList.append('drop ' + percent(layer['dropout']) +
                             ' -> ' + layer['activation'] +
                              '(' + str(layer['n_out']) + ')')

        graphStr = ' '.join(graphList)

        if self._supervised_cost:
            supCostStr = self.schema['supervised-cost']['type']
        else:
            supCostStr = "None"

        if self._unsupervised_cost:
            unsupCostStr = self.schema['unsupervised-cost']['type']
        else:
            unsupCostStr = "None"

        return ("Computational graph:\n" +
                " - description : " + self.schema['description'] + '\n' +
                " - graph layout: " + graphStr                   + '\n' + 
                " - cost(sup.)  : " + supCostStr                 + '\n' + 
                " - cost(unsup.): " + unsupCostStr               + '\n' )





