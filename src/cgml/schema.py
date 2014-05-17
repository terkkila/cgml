
import theano.tensor as T
from cgml.activations import activationMap

allowedGraphs = ['classifier','regressor','autoencoder','reinforcement-learner']

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

        if layer['activation'] == 'conv2d' and i > 0:
            raise Exception("Convolution operator can currently only appear in the first layer")
        
        if layer['activation'] == 'conv2d' and not layer.get('n_filters'):
            raise Exception("When using activation 'conv2d', the number of filters 'n_filters' needs to be specified")

        if layer['activation'] == 'conv2d' and layer['n_filters'] != 1:
            raise Exception("When using activation 'conv2d', the number of filters 'n_filters' is currently limited to 1")

        if layer['activation'] == 'conv2d' and not layer.get('filter_dim'):
            raise Exception("When using activation 'conv2d', the filter dimensionality 'filter_dim' needs to be specified")

        if layer.get('dropout') == None:
            raise Exception("Layer " + str(layer) + " is missing 'dropout'")

        if 0 > layer['dropout'] or layer['dropout'] > 1:
            raise Exception("Layer " + str(layer) + "does not have dropout in 0..1")
