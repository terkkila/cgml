
import argparse
from classifiers import LogRegClassifier,MultiLayerPerceptronClassifier
from autoencoders import AutoEncoder

class Callable:

    def __repr__(self):
        return self.model_.__name__

    def __str__(self):
        return self.__repr__()
        
    def __call__(self,*args,**kwargs):
        return self.model_(*args,**kwargs)


class Model(Callable):

    def __init__(self,modelStr):
        if modelStr == 'logreg':
            self.model_ = LogRegClassifier
        elif modelStr == 'mlp':
            self.model_ = MultiLayerPerceptronClassifier
        elif modelStr == 'ae':
            self.model_ = AutoEncoder
        else:
            self.model_ = None

class ModelAction(argparse.Action):

    choices = ['logreg','mlp','ae']
    default = Model('logreg')
    
    def __call__(self, parser, namespace, choice, option_string = None):

        model = Model(choice) 
        setattr(namespace,self.dest,model) 

