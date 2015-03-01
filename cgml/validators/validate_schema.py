
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
            _validate_convolution_layer(layer)
        else:
            convLayers.append(False)
            _validate_regular_layer(layer)

        if layer.get('dropout') == None:
            raise Exception("Layer " + str(layer) + " is missing 'dropout'")

        if 0 > layer['dropout'] or layer['dropout'] > 1:
            raise Exception("Layer " + str(layer) + "does not have dropout in 0..1")
