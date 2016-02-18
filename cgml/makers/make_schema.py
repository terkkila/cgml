
from cgml.constants import SCHEMA_IDS as SID
from cgml.validators import validateSchema

def makeSchema(n_in=None,
               n_out=None,
               nLayers=1,
               inputDropRate=2,
               modelType=None,
               costFunction=None,
               activationFunction="tanh",
               useDropout=True):

    last_n_in = n_in

    layers = []
    
    inputDropoutRate = (0.2 if useDropout else 0.0)
    dropoutRate = (0.5 if useDropout else 0.0)

    if nLayers > 1:

        for i in range(nLayers - 1):

            curr_n_out = int(round(last_n_in / inputDropRate))

            if curr_n_out <= n_out:
                curr_n_out = n_out
            
            layer = {SID.LAYER_NAME: "hidden{0}".format(i),
                     SID.LAYER_N_IN: last_n_in,
                     SID.LAYER_N_OUT: curr_n_out,
                     SID.LAYER_ACTIVATION: activationFunction,
                     SID.LAYER_DROPOUT: (inputDropoutRate if i == 0 else dropoutRate)}
            
            layers.append(layer)

            last_n_in = curr_n_out
        
    # No dropout with nLayers == 1, which is linear model
    lastLayer = {SID.LAYER_NAME: "output",
                 SID.LAYER_N_IN: last_n_in,
                 SID.LAYER_N_OUT: n_out,
                 SID.LAYER_ACTIVATION: "linear",
                 SID.LAYER_DROPOUT: (0.0 if nLayers == 1 else dropoutRate)} 

    layers.append(lastLayer)
    
    schema = {SID.DESCRIPTION: "schema by maker",
              SID.MODEL_TYPE: modelType,
              SID.SUPERVISED_COST: {SID.COST_NAME: "output",
                                    SID.COST_TYPE: costFunction},
              SID.GRAPH: layers}

    validateSchema(schema)

    return schema
