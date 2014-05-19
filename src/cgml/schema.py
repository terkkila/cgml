
import math
import theano.tensor as T
from cgml.activations import activationMap

allowedGraphs = ['classifier','regressor','autoencoder','reinforcement-learner']

def isQuadratic(x):
    return math.sqrt(x) % 1 == 0 

def validateSchema(schema):

    if not schema.get('description'):
        raise Exception("Schema does not have field 'description'")

    if not schema.get('type'):
        raise Exception("Schema does not have field 'type'")

    if schema['type'] not in allowedGraphs:
        raise Exception("Schema is not of allowed type: " + str(allowedGraphs))

    if not schema.get('randomInit'):
        raise Exception("Schema does not have field 'randomInit'")
    
    if not schema.get("graph"):
        raise Exception("Schema does not have field 'graph'")

    nLayers = len(schema['graph'])

    if nLayers == 0:
        raise Exception("Graph in schema has no layers")

    for i,layer in zip(xrange(nLayers),schema['graph']):

        if not layer.get('n_in'):
            raise Exception("Layer " + str(layer) + " is missing 'n_in'")

        if layer['n_in'] <= 0:
            raise Exception("Layer " + str(layer) + " must have positive number of 'n_in'")

        if not layer.get('n_out'):
            raise Exception("Layer " + str(layer) + " is missing 'n_out'")

        if layer['n_out'] <= 0:
            raise Exception("Layer " + str(layer) + " must have positive number of 'n_out'")

        if not layer.get('activation'):
            raise Exception("Layer " + str(layer) + " is missing 'activation'")

        if layer['activation'] not in activationMap.keys():
            raise Exception("Activation of the layer " + str(layer) + " is not of allowed type: " + 
                            str(activationMap.keys()))

        if layer['activation'] == 'conv2d':

            whenConv = "When using activation 'conv2d'"

            if not layer.get('n_filters'):
                raise Exception(whenConv+", the number of filters 'n_filters' " +
                                "needs to be specified")
            
            if not layer.get('filter_width'):
                raise Exception(whenConv+", the filter dimensionality 'filter_width' " +
                                "needs to be specified")

            if not isQuadratic(layer['n_in']):
                raise Exception(whenConv+", input must be mappable to a square shape")

            imWidth = math.sqrt(layer['n_in'])

            if imWidth < layer['filter_width']:
                raise Exception(whenConv+", the filter cannot have greater width " +
                                "than that of input")

            if not layer.get('subsample'):
                raise Exception(whenConv+", subsample needs to be specified")

            if (type(layer['subsample']) != list or
                len(layer['subsample']) != 2 or
                type(layer['subsample'][0]) != int or
                type(layer['subsample'][1]) != int):
                raise Exception(whenConv+", subsample should be a 2D tuple of integers")

            if layer['subsample'][0] != layer['subsample'][1]:
                raise Exception(whenConv+", subsampling both dimensions should be the same")

            if ( layer['n_out'] != layer['n_filters'] * 
                 (imWidth - layer['filter_width'] + 1) ** 2 / 
                 (layer['subsample'][0]*layer['subsample'][1]) ):
                raise Exception("When using activation 'conv2d', the output size " +
                                "should be equal to: 'n_filters*(im_width - filter_width + 1)^2 / " +
                                "(subsample[0]*subsample[1])'")

        if layer.get('dropout') == None:
            raise Exception("Layer " + str(layer) + " is missing 'dropout'")

        if 0 > layer['dropout'] or layer['dropout'] > 1:
            raise Exception("Layer " + str(layer) + "does not have dropout in 0..1")
