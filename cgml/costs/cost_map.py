
from cgml.costs import *

costMap = {
    'negative-log-likelihood': nllCost,
    'squared-error': sqerrCost,
    'cross-entropy': crossEntCost,
    'absolute-error': absCost,
    'absolute-percentage-error': apeCost,
    'huber-error': huberCost
    }
