
import types
import re
import copy

def _is_function(f):
    return isinstance(f, types.FunctionType)

def _make_camelcase(name):

    r = re.compile('-.')

    camelName = copy.deepcopy(name)
    
    for elem in r.findall(name):
        camelName = name.replace(elem,elem[1].upper())

    return camelName
