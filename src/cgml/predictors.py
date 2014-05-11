
import theano
import theano.tensor as T

class OnlinePredictor(object):

    def __init__(self,
                 x = None,
                 model = None):

        if model.type == 'classifier':
        
            self.predict_model = theano.function( inputs = [x],
                                                  outputs = T.argmax(model.output,
                                                                     axis = 1) )

        if model.type == 'autoencoder':

            self.predict_model = theano.function( inputs = [x],
                                                  outputs = model.output )

            encodeLayerIdx = len(model.layers) / 2 - 1

            self.encode_model = theano.function( inputs = [x],
                                                 outputs = model.layers[encodeLayerIdx].output )

            
    def predict(self,x):

        # Obtain predictions, which may be a vector
        return self.predict_model(x)

        # If the length of the prediction is 1, we can unwrap that
        #return ( y_pred if len(y_pred) > 1 else y_pred[0] )


    def encode(self,x):

        return self.encode_model(x)








