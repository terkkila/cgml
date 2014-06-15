
import theano
import theano.tensor as T

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


class AdaDelta(object):

    def __init__(self, 
                 cost = None,
                 params = None,
                 decay = None,
                 epsilon = None,
                 learnRate = None,
                 momentum = None):

        if cost == None:
            raise Exception("cost is missing!")

        if params == None:
            raise Exception("params is missing!")

        if epsilon == None:
            epsilon = 1e-3
            #raise Exception("epsilon is missing!")

        if decay == None:
            decay = 0.95
            #raise Exception("decay is missing!")

        ## LEARNRATE and MOMENTUM are not used
        
        self.gms = []
        self.sms = []
        #self.prevDeltas = []

        for param in params:
            self.gms.append( theano.shared(value = param.zeros_like().eval()) )
            self.sms.append( theano.shared(value = param.zeros_like().eval()) )
            #self.prevDeltas.append( theano.shared(value = param.zeros_like().eval()) )

        self.updates = []
        
        for param,gm,sm in zip(params,
                               self.gms,
                               self.sms):


            grad = T.grad(cost,param)

            gm_new = decay * gm + ( 1 - decay ) * grad ** 2

            self.updates.append( (gm,gm_new) )
            

            delta = - T.sqrt(sm + epsilon) / T.sqrt(gm_new + epsilon) * grad

            param_new = param + delta

            self.updates.append( (param,param_new) )


            sm_new = decay * sm + ( 1 - decay ) * delta
            
            self.updates.append( (sm,sm_new) )
