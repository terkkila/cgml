
from layers import Layer

class AutoEncoder(object):

    def __init__(self, x = None, n_in = None):

        self.encoder = Layer(input = x,
                             n_in = n_in,
                             n_out = 2,
                             activation = None)

        self.decoder = Layer(input = self.encoder.output,
                             n_in = 2,
                             n_out = n_in,
                             activation = None,
                             W = self.encoder.W.T,
                             b = self.encoder.b.T)

        self.params = self.encoder.params

        self.output = self.decoder.output




















