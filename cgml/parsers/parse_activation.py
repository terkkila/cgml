
from .base import _is_function

import cgml.activations

def parseActivation(s):

    activation = cgml.activations.__dict__.get(s,None)
    
    if activation is None:
        raise Exception("Could not parse activation function for identifier '" + s + "'")

    if not _is_function(activation):
        raise Exception("Activation with identifier '"+ s +"' is not a function")

    return activation
