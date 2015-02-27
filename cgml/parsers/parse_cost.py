
from .base import _is_function
from .base import _make_camelcase

import cgml.costs

def parseCost(s):

    cost = cgml.costs.__dict__.get( _make_camelcase(s) , None)

    if cost is None:
        raise Exception("Cloud not parse cost function with identifier '" + s + "'")

    if not _is_function(cost):
        raise Exception("Cost with identifier '"+ s +"' is not a function")

    return cost
