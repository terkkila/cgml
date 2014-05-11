
import theano

class OnlineTrainer(object):
    
    def __init__(self,
                 x = None,
                 y = None,
                 cost = None,
                 optimizer = None):

        if x == None:
            raise Exception("Input variabble x needs to be specified for OnlineTrainer")

        # Assume the underlying model is not an autoencoder
        self._isAutoEncoder = False

        # However, if target symbol is not given, then we assume the model
        # is in fact autoencoder and build the update rule accordingly
        if y == None:
            
            self._isAutoEncoder = True
            
            self.update_autoencoder_model = theano.function(inputs  = [x],
                                                            outputs = cost,
                                                            updates = optimizer.updates)
            
        else:
            
            self.update_model = theano.function(inputs  = [x,y],
                                                outputs = cost,
                                                updates = optimizer.updates)
   
            

        # Collect all costs in a vector as model gets updated
        self.costVec = []
        
    # Update model
    def update(self,
               x_train,
               y_train):

        # If target is not given, it is possible that the underlying model is an autoencoder
        if y_train == None:
            
            if self._isAutoEncoder:
        
                self.costVec.append( self.update_autoencoder_model(x_train) )
                
            else:
                
                raise Exception("Target needs to be given when updating the model!")
            
        else:
            
            self.costVec.append( self.update_model(x_train,y_train) )


        











