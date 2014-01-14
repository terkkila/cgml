

import argparse
from classifiers import LogRegClassifier,MultiLayerPerceptronClassifier

class Model(object):

    def __init__(self,modelStr):
        if modelStr == 'logreg':
            self.model_ = LogRegClassifier
        elif modelStr == 'mlp':
            self.model_ = MultiLayerPerceptronClassifier
        else:
            self.model_ = None

    def __repr__(self):
        return self.model_.__name__

    def __str__(self):
        return self.__repr__()
        
    def __call__(self,*args,**kwargs):
        return self.model_(*args,**kwargs)

class ModelAction(argparse.Action):
    def __call__(self,parser,namespace,value,option_string=None):
        try:  #Catch the runtime error if it occurs.
           l = Model(value) 
        except RuntimeError as E:
           #Optional:  Print some other error here.  for example: `print E; exit(1)`  
           parser.error()

        setattr(namespace,self.dest,l) 
