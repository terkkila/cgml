
from cgml.layers import Layer,ConvolutionLayer
from cgml.activations import activationMap

def makeDropoutLayersFromSchema(x,schema,rng):

    schemaLayers = schema['graph']

    nLayers = len(schemaLayers)

    dropoutLayers = []
    branchDropoutLayer = None
    
    lastOutput = x
    lastNOut = schemaLayers[0]['n_in']

    for i in xrange(nLayers):

        currDropoutLayer = schema['graph'][i]

        isCurrConvLayer = currDropoutLayer['activation'] == 'conv2d'

        if not isCurrConvLayer:
            
            dropoutLayers.append( Layer(rng        = rng,
                                        input      = lastOutput,
                                        n_in       = lastNOut,
                                        n_out      = currDropoutLayer['n_out'],
                                        activation = activationMap[
                        currDropoutLayer['activation'] ],
                                        randomInit = True,
                                        dropout    = currDropoutLayer['dropout'],
                                        name       = currDropoutLayer.get('name',
                                                                          "unnamed")) )
            
            
        else:

            dropoutLayers.append( ConvolutionLayer(rng          = rng,
                                                   input        = lastOutput,
                                                   n_in         = lastNOut,
                                                   n_out        = currDropoutLayer['n_out'],
                                                   activation   = activationMap[
                        currDropoutLayer['activation'] ],
                                                   randomInit   = True,
                                                   dropout      = currDropoutLayer['dropout'],
                                                   filter_width = currDropoutLayer['filter_width'],
                                                   subsample    = currDropoutLayer['subsample'],
                                                   maxpool      = currDropoutLayer['maxpool'],
                                                   name         = currDropoutLayer.get('name',
                                                                                       "unnamed")))

        if currDropoutLayer.get('branch'):

            assert currDropoutLayer['branch'][0]['activation'] != 'conv2d'

            branchDropoutLayer = Layer(rng        = rng,
                                       input      = lastOutput,
                                       n_in       = lastNOut,
                                       n_out      = currDropoutLayer['branch'][0]['n_out'],
                                       activation = activationMap[
                    currDropoutLayer['branch'][0]['activation'] ],
                                       randomInit = True,
                                       dropout    = currDropoutLayer['branch'][0]['dropout'],
                                       name       = currDropoutLayer['branch'][0].get('name',
                                                                                      "unnamed"))

        lastOutput = dropoutLayers[-1].output
        lastNOut   = dropoutLayers[-1].n_out

    return dropoutLayers,branchDropoutLayer


def makeLayersFromDropoutLayers(x,
                                schema,
                                dropoutLayers,
                                branchDropoutLayer):

    nLayers = len(dropoutLayers)

    layers = []
    branchLayer = None

    lastOutput = x
    lastNOut = dropoutLayers[0].n_in

    graphHasBranch = False
    layerHasBranch = False

    for i in xrange(nLayers):

        
        layerHasBranch = (True if schema['graph'][i].get('branch') else False)

        if layerHasBranch:
            graphHasBranch = True

        activationStr = schema['graph'][i]['activation']

        currDropoutLayer = dropoutLayers[i]

        isCurrConvLayer = schema['graph'][i]['activation'] == 'conv2d'
        
        q = 1 - currDropoutLayer.dropout
        
        if not isCurrConvLayer:
            
            layers.append( Layer(rng = None,
                                 input = lastOutput,
                                 n_in  = lastNOut,
                                 n_out = currDropoutLayer.n_out,
                                 activation = currDropoutLayer.activation,
                                 W = currDropoutLayer.W * q,
                                 b = currDropoutLayer.b,
                                 dropout = 0,
                                 name = currDropoutLayer.name) )

            if layerHasBranch:
                branchLayer = Layer(rng   = None,
                                    input = lastOutput,
                                    n_in  = lastNOut,
                                    n_out = branchDropoutLayer.n_out,
                                    activation = branchDropoutLayer.activation,
                                    W = branchDropoutLayer.W * q,
                                    b = branchDropoutLayer.b,
                                    dropout = 0,
                                    name = branchDropoutLayer.name)

            
        else:

            layers.append( ConvolutionLayer(rng = None,
                                            input = lastOutput,
                                            n_in  = lastNOut,
                                            n_out = currDropoutLayer.n_out,
                                            activation = currDropoutLayer.activation,
                                            W = currDropoutLayer.W * q,
                                            dropout = 0,
                                            filter_width = currDropoutLayer.filter_width,
                                            subsample = currDropoutLayer.subsample,
                                            maxpool = currDropoutLayer.maxpool,
                                            name = currDropoutLayer.name) )
            
        lastOutput = layers[-1].output
        lastNOut   = layers[-1].n_out
 
    return layers,branchLayer
        

def parseGraphFromSchema(x,schema,rng):

    dropoutLayers,branchDropoutLayer = makeDropoutLayersFromSchema(x,
                                                                   schema,
                                                                   rng)

    layers,branchLayer = makeLayersFromDropoutLayers(x,
                                                     schema,
                                                     dropoutLayers,
                                                     branchDropoutLayer)
    
    if ( branchDropoutLayer != None and 
         branchLayer != None ):
        layers.append(branchLayer)
        dropoutLayers.append(branchDropoutLayer)

    return layers,dropoutLayers

