
import argparse
from classifiers import LogRegClassifier,MultiLayerPerceptronClassifier
from autoencoders import AutoEncoder
from costs import nllCost,sqerrCost

class Callable:

    def __repr__(self):
        return self.val_.__name__

    def __str__(self):
        return self.__repr__()
        
    def __call__(self,*args,**kwargs):
        return self.val_(*args,**kwargs)

class Model(Callable):
    
    def __init__(self,modelStr):
        if modelStr == 'logreg':
            self.val_ = LogRegClassifier
        elif modelStr == 'mlp':
            self.val_ = MultiLayerPerceptronClassifier
        elif modelStr == 'ae':
            self.val_ = AutoEncoder
        else:
            self.val_ = None

class Cost(Callable):
    
    def __init__(self,costStr):
        if costStr == 'nll':
            self.val_ = nllCost
        elif costStr == 'sqerr':
            self.val_ = sqerrCost
        else:
            self.val_ = None

            
class ModelAction(argparse.Action):

    choices = ['logreg','mlp','ae']
    default = None
    
    def __call__(self, parser, namespace, choice, option_string = None):

        model = Model(choice) 
        setattr(namespace, self.dest, model)

class CostAction(argparse.Action):

    choices = ['nll','sqerr']
    default = None

    def __call__(self, parser, namespace, choice, option_string = None):

        cost = Cost(choice)
        setattr(namespace, self.dest, cost)


def giveArgs():

    parser = argparse.ArgumentParser( description = 'Machine Learning with Computational Graphs.')
    
    parser.add_argument(
        '--model',
        choices  = ModelAction.choices,
        action   = ModelAction,
        help     = 'Model',
        dest     = 'Model',
        default  = ModelAction.default,
        required = True)
    
    parser.add_argument(
        '--cost',
        choices  = CostAction.choices,
        action   = CostAction,
        help     = 'Cost function to use',
        dest     = 'Cost',
        default  = CostAction.default,
        required = True)

    parser.add_argument(
        '--trainData',
        help     = 'Data for training',
        dest     = 'trainData',
        default  = None,
        required = True)

    parser.add_argument(
        '--testData',
        help     = 'Data for testing',
        dest     = 'testData',
        default  = None,
        required = True)

    parser.add_argument(
        '--learnRate',
        help = 'Learning rate for the stochastic gradient descent algorithm',
        type = float,
        dest = 'learnRate',
        default = None,
        required = True)
    
    args = parser.parse_args()
    
    return args










