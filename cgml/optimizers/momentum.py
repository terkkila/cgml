

class Momentum(object):
    
    def __init__(self, 
                 cost = None, 
                 params = None, 
                 learnRate = None,
                 momentum = None,
                 decay = None,
                 epsilon = None):

        if cost == None:
            raise Exception("cost is missing!")

        if params == None:
            raise Exception("params is missing!")

        if learnRate == None:
            raise Exception("learnRate is missing!")

        if momentum == None:
            raise Exception("momentum is missing!")

        ## EPSILON and DECAY are not used

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

