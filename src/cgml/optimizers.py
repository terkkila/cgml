
import theano.tensor as T

class MSGD(object):

    def __init__(self, cost = None, params = None, learnRate = None):

        self.updates = []
        for param in params:
            self.updates.append( (param, param - learnRate * T.grad(cost,param)) )







