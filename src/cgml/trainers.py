
import theano

class OnlineTrainer(object):
    
    def __init__(self,
                 x = None,
                 y = None,
                 cost = None,
                 optimizer = None):
        
        self.update_model = theano.function(inputs  = [x,y],
                                            outputs = cost,
                                            updates = optimizer.updates)

        self.costVec = []
        
    def update(self,x_train,y_train):
        self.costVec.append( self.update_model(x_train,y_train) )


        











