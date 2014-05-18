
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

            if i > 0:
                raise Exception("Convolution operator can currently only appear in the first layer")
        
            if not layer.get('n_filters'):
                raise Exception("When using activation 'conv2d', the number of filters 'n_filters' needs to be specified")
            
            if layer['n_filters'] != 1:
                raise Exception("When using activation 'conv2d', the number of filters 'n_filters' is currently limited to 1")

            if not layer.get('filter_width'):
                raise Exception("When using activation 'conv2d', the filter dimensionality 'filter_width' needs to be specified")

            if not isQuadratic(layer['n_in']):
                raise Exception("When using activation 'conv2d', input must be mappable to a square shape")

            imWidth = math.sqrt(layer['n_in'])

            if imWidth < layer['filter_width']:
                raise Exception("When using activation 'conv2d', the filter cannot have greater width than that of input")

            if layer['n_out'] != (imWidth - layer['filter_width'] + 1) ** 2:
                raise Exception("When using activation 'conv2d', the output size should be equal to: '(im_width - filter_width + 1)^2'")

        if layer.get('dropout') == None:
            raise Exception("Layer " + str(layer) + " is missing 'dropout'")

        if 0 > layer['dropout'] or layer['dropout'] > 1:
            raise Exception("Layer " + str(layer) + "does not have dropout in 0..1")
