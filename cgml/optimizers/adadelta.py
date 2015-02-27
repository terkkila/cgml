
import theano
import theano.tensor as T

class AdaDelta(object):

    def __init__(self, 
                 cost = None,
                 params = None,
                 epsilon = 1e-6,
                 decay = 0.95,
                 momentum = 0.9):

        if cost == None:
            raise Exception("cost is missing!")

        if params == None:
            raise Exception("params is missing!")

        self.gms = []
        self.sms = []

        # Store previous update deltas here
        self.prevDeltas = []

        for param in params:
            self.prevDeltas.append( theano.shared(value = param.zeros_like().eval()) )
            self.gms.append( theano.shared(value = param.zeros_like().eval()) )
            self.sms.append( theano.shared(value = param.zeros_like().eval()) )

        self.updates = []

        for param,prevDelta,gm,sm in zip(params,
                                         self.prevDeltas,
                                         self.gms,
                                         self.sms):

            momentumFactor = momentum * prevDelta

            grad = T.grad(cost,param)

            gm_new = decay * gm + ( 1 - decay ) * grad ** 2

            self.updates.append( (gm,gm_new) )
            
            delta = - T.sqrt(sm + epsilon) / T.sqrt(gm_new + epsilon) * grad

            # Assign update rule for previous delta
            self.updates.append( (prevDelta, delta) )

            param_new = param + delta + momentumFactor

            self.updates.append( (param,param_new) )

            sm_new = decay * sm + ( 1 - decay ) * delta ** 2
            
            self.updates.append( (sm,sm_new) )

    


        
