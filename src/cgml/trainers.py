
import theano

class OnlineTrainer(object):
    
    def __init__(self,
                 x = None,
                 y = None,
                 model = None,
                 cost = None,
                 optimizer = None,
                 verbose = False):
        
        self.x = x
        self.y = y
        self.model = model
        self.cost = cost
        self.optimizer = optimizer
        self.verbose = verbose
        
        self.update_model = theano.function(inputs  = [x,y],
                                            outputs = cost,
                                            updates = optimizer.updates)
        
    def update(self,x_train,y_train):
        self.update_model(x_train,y_train)


        











