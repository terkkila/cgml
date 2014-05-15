
import theano
import theano.tensor as T

class MSGD(object):
    
    def __init__(self, cost = None, 
                 params = None, 
                 learnRate = None,
                 momentum = None):

        # Store previous update deltas here
        self.prevDeltas = []

        # Initialize the previous update deltas to zero
        for param in params:
            self.prevDeltas.append( theano.shared(value = param.zeros_like().eval()) )

        # Collect all updates
        self.updates = []

        for param,prevDelta in zip(params,self.prevDeltas):

            # Calculate current update delta
            delta = learnRate * T.grad(cost,param)

            # Assign update rule for param
            self.updates.append( (param, 
                                  param - delta - momentum * prevDelta) )
            
            # Assign update rule for previous delta
            self.updates.append( (prevDelta, delta) )






