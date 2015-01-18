
import sys
import numpy as np
import theano
import theano.tensor as T
from cgml.graph_parsing import makeDropoutLayersFromSchema
from cgml.graph_parsing import makeLayersFromDropoutLayers
from cgml.graph_parsing import parseGraphFromSchema
from cgml.layers import Layer,ConvolutionLayer
from cgml.activations import activationMap
from cgml.costs import costMap
from cgml.optimizers import Momentum,AdaDelta
from cgml.schema import validateSchema
from cgml.io import ppf
from cgml.constants import DEFAULT_ADADELTA_EPSILON
from cgml.constants import DEFAULT_ADADELTA_DECAY
from cgml.constants import DEFAULT_ADADELTA_MOMENTUM
from cgml.constants import DEFAULT_MINI_BATCH_SIZE
import cPickle
import copy
from collections import OrderedDict

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

def percent(x):
    return '{0:3.2f}'.format(100*x) + '%'
    
class ComputationalGraph(object):

    def __init__(self,
                 schema = None,
                 log   = None,
                 epsilon = DEFAULT_ADADELTA_EPSILON,
                 decay = DEFAULT_ADADELTA_DECAY,
                 momentum = DEFAULT_ADADELTA_MOMENTUM,
                 seed = None,
                 supCostWeight = 1,
                 unsupCostWeight = 1):

        self.meta = None

        self.supCostWeight = supCostWeight
        self.unsupCostWeight = unsupCostWeight
        self.epsilon = epsilon
        self.decay = decay
        self.momentum = momentum

        # Run schema validator before we do anything
        validateSchema(schema)

        self.type = schema['type']

        # Input data is always a data matrix
        self.input = T.fmatrix('x')

        self.X_in_device = theano.shared( value = np.asarray( [[np.nan]],
                                                              dtype = theano.config.floatX ) )

        if ( schema.get("supervised-cost") and 
             schema["supervised-cost"]["name"] in ["class-out"] ):

            self.targetType = np.int

            # Symbolic output matrix
            self.output = T.lmatrix('y')

        else:

            self.targetType = theano.config.floatX
            
            # Symbolic output matrix
            self.output = T.fmatrix('y')
            
            
        self.y_in_device = theano.shared( value = np.asarray( [[0]],
                                                              dtype = self.targetType ) )

        self.seed = seed

        self.rng = np.random.RandomState(self.seed)

        # Schema to build the model from
        self.schema = schema

        if log:
            log.write('Loaded the following schema: ' +
                      str(self.schema) + '\n')

        # Parse layers from the schema. Input is needed to clamp
        # it with the first layer
        self.layers,self.dropoutLayers = parseGraphFromSchema(self.input,
                                                              self.schema,
                                                              self.rng)

        # Collect parameters of all the layers
        self.params = [param for layer in self.dropoutLayers
                       for param in layer.params]

        self.compile(log = log)

    def compile(self, log = None):

        if log:
            log.write("Compiling computational graph:\n")

        index = T.lscalar('index')

        miniBatchSize = T.lscalar('miniBatchSize')

        if log:
            log.write(" - Setting up and compiling outputs\n")
        self._setUpOutputs(self.input)

        if log:
            log.write(" - Setting up and compiling cost functions\n")
        self._setUpCostFunctions(self.input,
                                 self.output,
                                 self.supCostWeight,
                                 self.unsupCostWeight)

        if log:
            log.write(" - Setting up and compiling optimizers\n")
        self._setUpOptimizers(index,
                              miniBatchSize,
                              self.input,
                              self.output,
                              self.epsilon,
                              self.decay,
                              self.momentum)

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
        self.predict_probs = None

        for layer,dropoutLayer in zip(self.layers,self.dropoutLayers):

            # If we find a layer that has supervised cost associated with it,
            # we will bind the respective output variables to that layer
            if ( self.schema.get('supervised-cost') and 
                 self.schema['supervised-cost']['name'] == dropoutLayer.name ):

                self._supervised_dropout_output = dropoutLayer.output
                self._supervised_output = layer.output

                # To simplify the notation of the upcoming importance calculation
                # we rename the variables. 
                # - _s refers to a symbolic variable
                # - i refers to row, and j refers to column
                X_s = x
                Y_s = self._supervised_output
                
                # Calculate the gragient of the output wrt. to all inputs
                # Will yield a 3d matrix, one 2d matrix per sample in the minibatch
                # Dimensions in the 2d matrix correspond to variable (1st dim)
                # and output element in the vectorized target (2nd dim)
                g,upd = theano.map(lambda i: theano.map(lambda j: T.grad(Y_s[i,j],
                                                                         X_s).take([i],
                                                                                   axis=0).ravel(), 
                                                        sequences = [T.arange(Y_s.shape[1])]),
                                   sequences = [T.arange(Y_s.shape[0])])
                
                # Compiled function for importance calculation
                self._importance = theano.function( inputs = [x],
                                                    outputs = g,
                                                    updates = upd)
                
                # If the target is categorical, we take argmax of the predicted probabilities
                if self.targetType == np.int:
                    self.predict = theano.function( inputs = [x],
                                                    outputs = T.argmax(self._supervised_output,
                                                                   axis = 1).ravel() )
                    self.predict_probs = theano.function( inputs = [x],
                                                          outputs = self._supervised_output,
                                                          allow_input_downcast = True )
                
                else:
                    
                    self.predict = theano.function( inputs = [x],
                                                    outputs = self._supervised_output )


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


    def importance(self,x):

        # Will return a stack of 2d matrices, one matrix per sample in the minibatch
        rawImportance = self._importance(x)

        finalImportance = []

        # Loop through each 2d matrix in the minibatch
        for tmpImportance in rawImportance:

            # Assign name to each variable importance vector for vectorized outputs
            sampleImportance = OrderedDict([ (name,variableImportance.tolist()) 
                                             for name,variableImportance in zip(self.schema['names'],
                                                                                tmpImportance.T)])

            finalImportance.append(sampleImportance)

        return finalImportance
                
    def _setUpCostFunctions(self,
                            x,
                            y,
                            supCostWeight,
                            unsupCostWeight):
        
        self._unsupervised_cost = None
        self._supervised_cost = None
        self._hybrid_cost = None
        
        if self._supervised_dropout_output:
            cost = costMap[self.schema['supervised-cost']['type']]
            self._supervised_cost = supCostWeight * cost(self._supervised_dropout_output,y)
            self.supervised_cost = theano.function(inputs = [x,y],
                                                   outputs = self._supervised_cost)

        if self._unsupervised_dropout_output:
            cost = costMap[self.schema['unsupervised-cost']['type']]
            self._unsupervised_cost = unsupCostWeight * cost(self._unsupervised_dropout_output,x)
            self.unsupervised_cost = theano.function(inputs = [x],
                                                     outputs = self._unsupervised_cost)

    
        if self._supervised_cost and self._unsupervised_cost:
            self._hybrid_cost = self._supervised_cost + self._unsupervised_cost
            self.hybrid_cost = theano.function(inputs = [x,y],
                                               outputs = self._hybrid_cost)


    def setTrainDataOnDevice(self,X,y):

        self.X_in_device.set_value(X)
        self.y_in_device.set_value(y)


    def _setUpOptimizers(self,
                         index,
                         miniBatchSize,
                         x,
                         y,
                         epsilon,
                         decay,
                         momentum):

        Optimizer = AdaDelta

        self.supervised_update = None

        self.unsupervised_update = None

        self.hybrid_update = None

        if self._hybrid_cost:

            self.hybrid_optimizer = Optimizer(
                cost      = self._hybrid_cost,
                params    = self.params,
                epsilon   = epsilon,
                decay     = decay,
                momentum  = momentum)

            self.hybrid_update = theano.function(
                inputs  = [index,miniBatchSize],
                outputs = self._hybrid_cost,
                updates = self.hybrid_optimizer.updates,
                givens  = {x:self.X_in_device[index:(index+miniBatchSize)],
                           y:self.y_in_device[index:(index+miniBatchSize)]})

            return

        if self._supervised_cost:
            
            self.supervised_optimizer = Optimizer(
                cost      = self._supervised_cost,
                params    = self.params,
                epsilon   = epsilon,
                decay     = decay,
                momentum  = momentum)
            
            self.supervised_update = theano.function(
                inputs  = [index,miniBatchSize],
                outputs = self._supervised_cost,
                updates = self.supervised_optimizer.updates,
                givens  = {x:self.X_in_device[index:(index+miniBatchSize)],
                           y:self.y_in_device[index:(index+miniBatchSize)]})

            
            self.supervised_cost = theano.function(
                inputs  = [x,y],
                outputs = self._supervised_cost)

        if self._unsupervised_cost:
            
            self.unsupervised_optimizer = Optimizer(
                cost      = self._unsupervised_cost,
                params    = self.params,
                epsilon   = epsilon,
                decay     = decay,
                momentum  = momentum)
            
            self.unsupervised_update = theano.function(
                inputs  = [index,miniBatchSize],
                outputs = self._unsupervised_cost,
                updates = self.unsupervised_optimizer.updates,
                givens  = {x:self.X_in_device[index:(index+miniBatchSize)]})

            self.unsupervised_cost = theano.function(
                inputs  = [x],
                outputs = self._unsupervised_cost)

    def summarizeParams(self):

        for layer in self.layers:
            name = layer.name
            W,b = layer.weights()

            sys.stdout.write("{mean,min,max}(W),{mean,min,max}(b) = " + 
                             ', '.join(map(ppf,[np.mean(abs(W.flatten())),
                                               np.min(W.flatten()),
                                               np.max(W.flatten()),
                                               np.mean(abs(b.flatten())),
                                               np.min(b.flatten()),
                                               np.max(b.flatten())])) + " (" + name + ")\n")    

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

    def saveToFile(self,fileName):

        f = open(fileName,'wb')
    
        cPickle.dump(self,f,protocol=cPickle.HIGHEST_PROTOCOL)

    def getSchema(self):
        
        schemaWithWeights = copy.deepcopy(self.schema)

        for schemaLayerWithWeights,dropoutLayer,schemaLayer in zip(schemaWithWeights['graph'],
                                                                  self.dropoutLayers,
                                                                  self.schema['graph']):
            schemaLayerWithWeights['W'] = dropoutLayer.W.get_value()
            schemaLayerWithWeights['b'] = dropoutLayer.b.get_value()

        return schemaWithWeights


    def train(self,
              drTrain = None,
              x_valid = None,
              y_valid = None,
              miniBatchSize = DEFAULT_MINI_BATCH_SIZE,
              verbose = False,
              log = None):

        trainLog = {'batchIdx':[],
                    'trainCost':[],
                    'validCost':[]}

        nBatches = 0

        n = 0
        nTh = 100
        
        deviceBatchSize = drTrain.batchSize

        isSupCost = self.schema.get("supervised-cost")
        isUnSupCost = self.schema.get("unsupervised-cost")
        isHybridCost = isSupCost and isUnSupCost
        
        doValidation = (x_valid != None and y_valid != None)

        currMeanCost = 0.0
        
        for sampleIDs,x_train,y_train in drTrain:
            
            self.setTrainDataOnDevice(x_train,y_train)
        
            for i in xrange(deviceBatchSize/miniBatchSize):

                r = np.random.randint(deviceBatchSize-miniBatchSize)
            
                nBatches += 1
                n += 1 
            
                if isHybridCost:
                    currMeanCost += (self.hybrid_update(r,miniBatchSize) - currMeanCost) / n
            
                elif isSupCost:
                    currMeanCost += (self.supervised_update(r,miniBatchSize) - currMeanCost) / n
                elif isUnSupCost:
                    currMeanCost += (self.unsupervised_update(r,miniBatchSize) - currMeanCost) / n
                
                if n % nTh == 0 and log:
                    log.write('Batch ' + str(nBatches) + 
                                    ', avg. train cost ' + str(currMeanCost))

                    trainLog['trainCost'].append(currMeanCost)

                    if doValidation:
                        if isHybridCost:
                            validSupCost = self.supervised_cost(x_valid,y_valid)
                            validUnsupCost = self.unsupervised_cost(x_valid)
                            validHybCost = self.hybrid_cost(x_valid,y_valid)
                            yhat = self.predict(x_valid)
                            pMisClass = np.mean(yhat != y_valid)
                            log.write(', valid.sup.cost ' + str(validSupCost) +
                                            ', valid.unsup.cost ' + str(validUnsupCost) + 
                                            ', valid.hyb.cost ' + str(validHybCost) + 
                                            ', classification error ' + str(100*pMisClass) + "%")
                        
                        elif isSupCost:
                            validCost = self.supervised_cost(x_valid,y_valid)
                            yhat = self.predict(x_valid)
                            pMisClass = np.mean(yhat != y_valid)
                            log.write(', validation cost ' + str(validCost) + 
                                            ', classification error ' + str(100*pMisClass) + "%")
                            trainLog['validCost'].append(validCost)
                        elif isUnSupCost:
                            validCost = self.unsupervised_cost(x_valid)
                            log.write(', validation cost ' + str(validCost))
                    

                    log.write('\n')

                    n = 0

                    currMeanCost = 0.0

                    #self.summarizeParams()

        return trainLog

    @classmethod
    def loadFromFile(cls,fileName):
        return cPickle.load(open(fileName,'rb'))

