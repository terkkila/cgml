
import numpy as np

from cgml.constants import SCHEMA_IDS as SID
from cgml.parsers import parseCost,parseActivation

from .base import _validate_convolution_layer
from .base import _validate_regular_layer

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

    if schema[SID.MODEL_TYPE] not in SID.SUPPORTED_MODEL_TYPES:
        raise Exception("Model type '{0}'not in {1}".format(schema[SID.MODEL_TYPE],
                                                            SID.SUPPORTED_MODEL_TYPES))

    if schema.get(SID.TARGET_SCALING):
        if schema[SID.TARGET_SCALING].get(SID.TARGET_SCALING_MEAN,None) == None:
            raise Exception("Target scaling defined but "+
                            "missing '{0}'".format(SID.TARGET_SCALING_MEAN))
            
        if schema[SID.TARGET_SCALING].get(SID.TARGET_SCALING_STDEV,None) == None:
            raise Exception("Target scaling defined but "+
                            "missing '{0}'".format(SID.TARGET_SCAING_STDEV))
            
    if schema.get(SID.SUPERVISED_COST):
        if not schema[SID.SUPERVISED_COST].get(SID.COST_TYPE):
            raise Exception("Supervised cost has to have '{0}'".format(COST_TYPE))
        if not schema[SID.SUPERVISED_COST].get(SID.COST_NAME):
            raise Exception("Supervised cost has to have '{0}' ".format(SID.COST_NAME)
                            +"to match with layer name")

        try:
            parseCost(schema[SID.SUPERVISED_COST][SID.COST_TYPE])
        except:
            raise Exception("Could not parse cost function with identifier "+
                            "'{0}'".format(schema[SID.SUPERVISED_COST][SID.COST_TYPE]))

    if schema.get(SID.UNSUPERVISED_COST):
        if not schema[SID.UNSUPERVISED_COST].get(SID.COST_TYPE):
            raise Exception("Unsupervised cost has to have '{0}'".format(SID.COST_TYPE))
        if not schema[SID.UNSUPERVISED_COST].get(SID.COST_NAME):
            raise Exception("Unsupervised cost has to have "+
                            "'{0}' to match with layer name".format(SID.COST_NAME))
        try:
            parseCost(schema[SID.UNSUPERVISED_COST][SID.COST_TYPE])
        except:
            raise Exception("Could not parse cost function with identifier '" + 
                            schema[SID.UNSUPERVISED_COST][SID.COST_TYPE] + "'")

    if not schema.get(SID.GRAPH):
        raise Exception("Schema does not have field '{}'".format(SID.GRAPH))

    nLayers = len(schema[SID.GRAPH])

    if nLayers == 0:
        raise Exception("Graph in schema has no layers")

    if not schema.get(SID.FEATURE_NAMES):
        schema[SID.FEATURE_NAMES] = ['f{0}'.format(i) 
                                     for i in xrange(np.prod(schema[SID.GRAPH][0][SID.LAYER_N_IN]))]
    
    if np.prod(schema[SID.GRAPH][0][SID.LAYER_N_IN]) != len(schema[SID.FEATURE_NAMES]):
        raise Exception("'{0}' ".format(SID.FEATURE_NAMES)+
                        "does not have same length as the number of inputs")

    convLayers = []

    seenNames = set()

    for i,layer in zip(xrange(nLayers),schema[SID.GRAPH]):

        if not layer.get(SID.LAYER_N_IN):
            raise Exception("Layer " + str(layer) + " is missing '{0}'".format(SID.LAYER_N_IN))

        if not layer.get(SID.LAYER_N_OUT):
            raise Exception("Layer " + str(layer) + " is missing '{0}'".format(SID.LAYER_N_OUT))

        if not layer.get(SID.LAYER_NAME):
            raise Exception("Layer " + str(layer) + " is missing '{0}'".format(SID_LAYER_NAME))

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
            _validate_convolution_layer(layer)
        else:
            convLayers.append(False)
            _validate_regular_layer(layer)

        if layer.get('dropout') == None:
            raise Exception("Layer " + str(layer) + " is missing 'dropout'")

        if 0 > layer['dropout'] or layer['dropout'] > 1:
            raise Exception("Layer " + str(layer) + "does not have dropout in 0..1")
