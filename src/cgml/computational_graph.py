
import theano.tensor as T
from layers import Layer
import yaml

allowedGraphs = ['classifier','regressor','autoencoder','reinforcement-learner']

activationMap = {'linear':  None,
                 'sigmoid': T.nnet.sigmoid,
                 'tanh':    T.tanh,
                 'softmax': T.nnet.softmax}

def parseLayerStr(layerStr):

    try:
        elems      = layerStr.split(' ')
        activation = activationMap[elems[0]]
        n_out      = int(elems[1])
    except:
        raise Exception('Cannot parse layer in graph: ' + layerStr)

    return activation,n_out

    
def parseLayers(x,schema):

    layers = []

    n_in = schema['n_in']
    
    randomInit = schema['randomInit']
    

    activation,n_out = parseLayerStr(schema['graph'][0])
    
    layers.append( Layer(input = x,
                         n_in = n_in,
                         n_out = n_out,
                         activation = activation,
                         randomInit = randomInit) )
    
    for i in xrange(1,len(schema['graph'])):

        n_in       = layers[i-1].n_out

        activation,n_out = parseLayerStr(schema['graph'][i])

        layers.append( Layer(input      = layers[i-1].output,
                             n_in       = n_in,
                             n_out      = n_out,
                             activation = activation,
                             randomInit = randomInit) )

    if layers[-1].n_out != schema['n_out']:
        raise Exception('Dimensionality of the output of the last layer (' +
                        str(layers[-1].n_out) + ') does not match that ' +
                        'specified in the schema (' + str(schema['n_out']) +
                        ')')

    return layers
        
    
class ComputationalGraph(object):

    def __init__(self,x,fileName):

        self.input = x
        
        self.schema = yaml.load(open(fileName,'r'))

        self.n_in  = self.schema['n_in']
        self.n_out = self.schema['n_out']
        
        self.layers = parseLayers(self.input,self.schema)

        self.params = []
        
        for layer in self.layers:
            self.params += layer.params

        self.output = self.layers[-1].output


















