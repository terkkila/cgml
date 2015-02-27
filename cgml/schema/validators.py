
import math
import theano.tensor as T
from cgml.parsers import parseActivation
from cgml.costs import costMap
import sys
import numpy as np

def validateConvolutionLayer(layer):

    whenConv = "When using activation 'conv2d'"
    
    if not layer.get('filter_width'):
        raise Exception(whenConv+", the filter dimensionality 'filter_width' " +
                        "needs to be specified")

    if ( type(layer['filter_width']) != list or 
         len(layer['filter_width']) != 2 ):
        raise Exception(whenConv+", 'filter_width' must be a list with two integers: "+
                        "filter x- and y- dimensions")
        
    if ( type(layer['n_in']) != list or
         len(layer['n_in']) != 3 ):
        raise Exception(whenConv+", 'n_in' must be a list with three integers: "+
                        "number of filters and x- and y- dimensions")
    
    if not layer.get('subsample'):
        raise Exception(whenConv+", 'subsample' needs to be specified")

    if ( type(layer['subsample']) != list or
         len(layer['subsample']) != 2 or
         type(layer['subsample'][0]) != int or
         type(layer['subsample'][1]) != int ):
        raise Exception(whenConv+", 'subsample' must be a list with two integers: "+
                        "amount of subsampling in x- and y- dimensions")

    if not layer.get('maxpool'):
        raise Exception(whenConv+", 'maxpool' needs to be specified")

    if ( type(layer['maxpool']) != list or
         len(layer['maxpool']) != 2 or
         type(layer['maxpool'][0]) != int or
         type(layer['maxpool'][1]) != int ):
        raise Exception(whenConv+", 'maxpool' must be a list with two integers: "+
                        "amount of maxpoolin in x- and y- dimensions")
    

    if ( type(layer['n_out']) != list or
         len(layer['n_out']) != 3 ):
        raise Exception(whenConv+", 'n_out' must be a list with three integers: "+
                        "number of filters and x- and y- dimensions")

    if ( layer['n_out'][1] != (layer['n_in'][1] - layer['filter_width'][0] + 1) / 
         (layer['subsample'][0] * layer['maxpool'][0]) ):
        raise Exception(whenConv+", 1st output dimensions should be "+
                        "'(n_in - filter_width + 1)/subsample'")

    if ( layer['n_out'][2] != (layer['n_in'][2] - layer['filter_width'][1] + 1) / 
         (layer['subsample'][1] * layer['maxpool'][1]) ):
        raise Exception(whenConv+", 2nd output dimensions should be "+
                        "'(n_in - filter_width + 1)/(subsample*maxpool)'")

    if layer.get('branch'):
        raise Exception(whenConv+", 'branch' cannot be set")


def validateRegularLayer(layer):

    if layer['n_in'] <= 0:
        raise Exception("Layer " + str(layer) + " must have positive number of 'n_in'")

    if layer['n_out'] <= 0:
        raise Exception("Layer " + str(layer) + " must have positive number of 'n_out'")


def validateSchema(schema):

    if not schema.get('description'):
        raise Exception("Schema does not have field 'description'")

    nCostDefs = ( (1 if schema.get('supervised-cost') else 0) + 
                  (1 if schema.get('unsupervised-cost') else 0) )

    if nCostDefs == 0:
        raise Exception("At least one of the following costs should be defined: " + 
                        "'supervised-cost', or 'unsupervised-cost'")
    
    if not schema.get("type"):
        raise Exception("Schema does not have field 'type'")

    supportedModelTypes = ["classification","regression","autoencoder","supervised-autoencoder"]

    if schema["type"] not in supportedModelTypes:
        raise Exception("Model type not in " + str(supportedModelTypes))

    if schema.get("target-scaling"):
        if schema["target-scaling"].get("mean",None) == None:
            raise Exception("Target scaling defined but missing 'mean'")
            
        if schema["target-scaling"].get("stdev",None) == None:
            raise Exception("Target scaling defined but missing 'stdev'")
            
    if schema.get('supervised-cost'):
        if not schema['supervised-cost'].get('type'):
            raise Exception("Supervised cost has to have 'type'")
        if not schema['supervised-cost'].get('name'):
            raise Exception("Supervised cost has to have 'name' to match with layer name")
        if schema['supervised-cost']['type'] not in costMap.keys():
            raise Exception("The type of supervised cost has to be in: " + str(costMap.keys()))

    if schema.get('unsupervised-cost'):
        if not schema['unsupervised-cost'].get('type'):
            raise Exception("Unsupervised cost has to have 'type'")
        if not schema['unsupervised-cost'].get('name'):
            raise Exception("Unsupervised cost has to have 'name' to match with layer name")
        if schema['unsupervised-cost']['type'] not in costMap.keys():
            raise Exception("The type of unsupervised cost has to be in: " + str(costMap.keys()))

    if not schema.get('graph'):
        raise Exception("Schema does not have field 'graph'")

    nLayers = len(schema['graph'])

    if nLayers == 0:
        raise Exception("Graph in schema has no layers")

    if not schema.get('names'):
        schema['names'] = ['f{0}'.format(i) for i in xrange(np.prod(schema['graph'][0]['n_in']))]
    
    if np.prod(schema['graph'][0]['n_in']) != len(schema['names']):
        raise Exception("'names' does not have same length as the number of inputs")

    convLayers = []

    seenNames = set()

    for i,layer in zip(xrange(nLayers),schema['graph']):

        if not layer.get('n_in'):
            raise Exception("Layer " + str(layer) + " is missing 'n_in'")

        if not layer.get('n_out'):
            raise Exception("Layer " + str(layer) + " is missing 'n_out'")

        if not layer.get('name'):
            raise Exception("Layer " + str(layer) + " is missing 'name'")

        if layer['name'] in seenNames:
            raise Exception("Seeing same layer name twice: " + layer['name'])

        seenNames.add(layer['name'])

        if not layer.get('activation'):
            raise Exception("Layer " + str(layer) + " is missing 'activation'")

        try:
            activation = parseActivation(layer['activation'])
        except Exception,e:
            raise Exception("Activation '" + layer['activation'] + "' could not be parsed")

        if layer['activation'] == 'conv2d':
            convLayers.append(True)
            if convLayers[0] == False:
                raise Exception("Convolution layers must start from the beginning")
            if i > 0 and convLayers[i-1] == False:
                raise Exception("Convolution layers must be following each other")
            validateConvolutionLayer(layer)
        else:
            convLayers.append(False)
            validateRegularLayer(layer)

        if layer.get('dropout') == None:
            raise Exception("Layer " + str(layer) + " is missing 'dropout'")

        if 0 > layer['dropout'] or layer['dropout'] > 1:
            raise Exception("Layer " + str(layer) + "does not have dropout in 0..1")
