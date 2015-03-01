
from cgml.constants import SCHEMA_IDS as SID
from cgml.validators import validateSchema

def makeSchema(n_in = None,
               n_out = None,
               modelType = None,
               costFunction = None):

    
    schema = {SID.DESCRIPTION: "schema by maker",
              SID.MODEL_TYPE: modelType,
              SID.SUPERVISED_COST: {SID.COST_NAME: "output",
                                    SID.COST_TYPE: costFunction},
              SID.GRAPH: [{SID.LAYER_NAME: "output",
                           SID.LAYER_N_IN: n_in,
                           SID.LAYER_N_OUT: n_out,
                           SID.LAYER_ACTIVATION: "linear",
                           SID.LAYER_DROPOUT: 0.0}]}

    validateSchema(schema)

    return schema
