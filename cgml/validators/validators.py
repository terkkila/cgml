
import math
import theano.tensor as T
from cgml.parsers import parseActivation,parseCost
import sys
import numpy as np

from cgml.constants import SCHEMA_IDS as SID

def validateConvolutionLayer(layer):

    whenConv = "When using activation 'conv2d'"
    
    if not layer.get(SID.FILTER_WIDTH):
        raise Exception(whenConv+", the filter dimensionality 'filter_width' " +
                        "needs to be specified")

    if ( type(layer[SID.FILTER_WIDTH]) != list or 
         len(layer[SID.FILTER_WIDTH]) != 2 ):
        raise Exception(whenConv+", 'filter_width' must be a list with two integers: "+
                        "filter x- and y- dimensions")
        
    if ( type(layer[SID.N_IN]) != list or
         len(layer[SID.N_IN]) != 3 ):
        raise Exception(whenConv+", 'n_in' must be a list with three integers: "+
                        "number of filters and x- and y- dimensions")
    
    if not layer.get(SID.SUBSAMPLE):
        raise Exception(whenConv+", 'subsample' needs to be specified")

    if ( type(layer[SID.SUBSAMPLE]) != list or
         len(layer[SID.SUBSAMPLE]) != 2 or
         type(layer[SID.SUBSAMPLE][0]) != int or
         type(layer[SID.SUBSAMPLE][1]) != int ):
        raise Exception(whenConv+", 'subsample' must be a list with two integers: "+
                        "amount of subsampling in x- and y- dimensions")

    if not layer.get(SID.MAX_POOL):
        raise Exception(whenConv+", 'maxpool' needs to be specified")

    if ( type(layer[SID.MAX_POOL]) != list or
         len(layer[SID.MAX_POOL]) != 2 or
         type(layer[SID.MAX_POOL][0]) != int or
         type(layer[SID.MAX_POOL][1]) != int ):
        raise Exception(whenConv+", 'maxpool' must be a list with two integers: "+
                        "amount of maxpoolin in x- and y- dimensions")
    

    if ( type(layer[SID.N_OUT]) != list or
         len(layer[SID.N_OUT]) != 3 ):
        raise Exception(whenConv+", '"+SID.N_OUT+"' must be a list with three integers: "+
                        "number of filters and x- and y- dimensions")

    if ( layer[SID.N_OUT][1] != (layer[SID.N_IN][1] - layer[SID.FILTER_WIDTH][0] + 1) / 
         (layer[SID.SUBSAMPLE][0] * layer[SID.MAX_POOL][0]) ):
        raise Exception(whenConv+", 1st output dimensions should be "+
                        "'(n_in - filter_width + 1)/subsample'")

    if ( layer[SID.N_OUT][2] != (layer[SID.N_IN][2] - layer[SID.FILTER_WIDTH][1] + 1) / 
         (layer[SID.SUBSAMPLE][1] * layer[SID.MAX_POOL][1]) ):
        raise Exception(whenConv+", 2nd output dimensions should be "+
                        "'(n_in - filter_width + 1)/(subsample*maxpool)'")

    if layer.get(SID.BRANCH):
        raise Exception(whenConv + ", '" + SID.BRANCH + "' cannot be set")


def validateRegularLayer(layer):

    if layer[SID.N_IN] <= 0:
        raise Exception("Layer " + str(layer) + " must have positive number of " + SID.N_IN)

    if layer[SID.N_OUT] <= 0:
        raise Exception("Layer " + str(layer) + " must have positive number of " + SID.N_OUT)


def validateSchema(schema):

    if not schema.get(SID.DESCRIPTION):
        raise Exception("Schema does not have field '{0}'".format(SID.DESCRIPTION))

    nCostDefs = ( (1 if schema.get(SID.SUPERVISED_COST) else 0) + 
                  (1 if schema.get(SID.UNSUPERVISED_COST) else 0) )
    
    if nCostDefs == 0:
        raise Exception("At least one of the following costs should be defined: " + 
                        "'{0}', or '{1}'".format(SID.SUPERVISED_COST,SID.UNSUPERVISED_COST))
    
    if not schema.get(SID.MODEL_TYPE):
        raise Exception("Schema does not have field '{0}'".format(SID.MODEL_TYPE))

    supportedModelTypes = ["classification","regression","autoencoder","supervised-autoencoder"]

    if schema[SID.MODEL_TYPE] not in SID.SUPPORTED_MODEL_TYPES:
        raise Exception("Model type '{0}'not in {1}".format(schema[SID.MODEL_TYPE],
                                                            SID.SUPPORTED_MODEL_TYPES))

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

        try:
            parseCost(schema['supervised-cost']['type'])
        except:
            raise Exception("Could not parse cost function with identifier '" + schema['supervised-cost']['type'] + "'")

    if schema.get('unsupervised-cost'):
        if not schema['unsupervised-cost'].get('type'):
            raise Exception("Unsupervised cost has to have 'type'")
        if not schema['unsupervised-cost'].get('name'):
            raise Exception("Unsupervised cost has to have 'name' to match with layer name")
        try:
            parseCost(schema['unsupervised-cost']['type'])
        except:
            raise Exception("Could not parse cost function with identifier '" + schema['supervised-cost']['type'] + "'")

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
