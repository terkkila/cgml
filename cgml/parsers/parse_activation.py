
import cgml.activations

def parseActivation(s):

    activation = cgml.activations.__dict__.get(s,None)
    
    if activation is None:
        raise Exception("Could not find activation function for identifier '" + s + "'")

    return activation
