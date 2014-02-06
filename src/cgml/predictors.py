
import theano
import theano.tensor as T

class OnlinePredictor(object):

    def __init__(self,
                 x = None,
                 model = None):

        self.predict_model = theano.function( inputs = [x],
                                              outputs = T.argmax(model.output,
                                                                 axis = 1) )

    def predict(self,x):

        # Obtain predictions, which may be a vector
        y_pred = self.predict_model(x)

        # If the length of the prediction is 1, we can unwrap that
        if len(y_pred) == 1:
            y_pred = y_pred[0]

        # Return prediction
        return y_pred








