
import math
import theano.tensor as T
import sys
import numpy as np

from cgml.parsers import parseActivation,parseCost
from cgml.constants import SCHEMA_IDS as SID

def _validate_convolution_layer(layer):

    whenConv = "When using activation 'conv2d'"
    
    if not layer.get(SID.FILTER_WIDTH):
        raise Exception(whenConv+", the filter dimensionality 'filter_width' " +
                        "needs to be specified")

    if ( type(layer[SID.FILTER_WIDTH]) != list or 
         len(layer[SID.FILTER_WIDTH]) != 2 ):
        raise Exception(whenConv+", 'filter_width' must be a list with two integers: "+
                        "filter x- and y- dimensions")
        
    if ( type(layer[SID.LAYER_N_IN]) != list or
         len(layer[SID.LAYER_N_IN]) != 3 ):
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
    

    if ( type(layer[SID.LAYER_N_OUT]) != list or
         len(layer[SID.LAYER_N_OUT]) != 3 ):
        raise Exception(whenConv+", '"+SID.LAYER_N_OUT+"' must be a list with three integers: "+
                        "number of filters and x- and y- dimensions")

    if ( layer[SID.LAYER_N_OUT][1] != (layer[SID.LAYER_N_IN][1] - layer[SID.FILTER_WIDTH][0] + 1) / 
         (layer[SID.SUBSAMPLE][0] * layer[SID.MAX_POOL][0]) ):
        raise Exception(whenConv+", 1st output dimensions should be "+
                        "'(n_in - filter_width + 1)/subsample'")

    if ( layer[SID.LAYER_N_OUT][2] != (layer[SID.LAYER_N_IN][2] - layer[SID.FILTER_WIDTH][1] + 1) / 
         (layer[SID.SUBSAMPLE][1] * layer[SID.MAX_POOL][1]) ):
        raise Exception(whenConv+", 2nd output dimensions should be "+
                        "'(n_in - filter_width + 1)/(subsample*maxpool)'")

    if layer.get(SID.BRANCH):
        raise Exception(whenConv + ", '" + SID.BRANCH + "' cannot be set")


def _validate_regular_layer(layer):

    if layer[SID.LAYER_N_IN] <= 0:
        raise Exception("Layer " + str(layer) + " must have positive number of " + SID.LAYER_N_IN)

    if layer[SID.LAYER_N_OUT] <= 0:
        raise Exception("Layer " + str(layer) + " must have positive number of " + SID.LAYER_N_OUT)
