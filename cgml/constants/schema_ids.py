

class SCHEMA_IDS(object):

    GRAPH = "graph"

    LAYER_NAME = "name"

    FEATURE_NAMES = "names"

    FILTER_WIDTH = "filter_width"
    DESCRIPTION = "description"
    N_IN = "n_in"
    N_OUT = "n_out"

    SUBSAMPLE = "subsample"
    MAX_POOL = "maxpool"

    BRANCH = "branch"

    SUPERVISED_COST = "supervised-cost"
    UNSUPERVISED_COST = "unsupervised-cost"

    COST_TYPE = "type"
    COST_NAME = "name"

    MODEL_TYPE = "type"

    CLASSIFICATION_MODEL_TYPE = "classification"
    REGRESSION_MODEL_TYPE = "regression"
    AUTOENCODER_MODEL_TYPE = "autoencoder"
    SUPERVISED_AUTOENCODER_MODEL_TYPE = "supervised-autoencoder"

    SUPPORTED_MODEL_TYPES = [CLASSIFICATION_MODEL_TYPE,
                             REGRESSION_MODEL_TYPE,
                             AUTOENCODER_MODEL_TYPE,
                             SUPERVISED_AUTOENCODER_MODEL_TYPE]


    TARGET_SCALING = "target-scaling"
    TARGET_SCALING_MEAN = "mean"
    TARGET_SCALING_STDEV = "stdev"
