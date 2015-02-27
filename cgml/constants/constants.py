
from enum import Enum

DEFAULT_DEVICE_BATCH_SIZE = 1000
DEFAULT_MINI_BATCH_SIZE = 10
DEFAULT_MINI_BATCH_INCREMENT = 0
DEFAULT_ADADELTA_EPSILON = 1e-6
DEFAULT_ADADELTA_DECAY = 0.1
DEFAULT_ADADELTA_MOMENTUM = 0.0

class TARGET_TYPE(object):

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    
    #@classmethod
    #def values(cls):
    #    return dir(cls)
