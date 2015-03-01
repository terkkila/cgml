

from nose.tools import raises
from cgml.makers import makeSchema
from cgml.validators import validateSchema

@raises(Exception)
def test_make_bad_regression_schema():
    # Typo in costFunction
    schema = makeSchema(n_in = 100, 
                        n_out = 5, 
                        modelType = "regression", 
                        costFunction = "square-error")
    

def test_make_good_regression_schema():
    schema = makeSchema(n_in = 100, 
                        n_out = 5, 
                        modelType = "regression", 
                        costFunction = "squared-error")

    validateSchema(schema)

@raises(Exception)
def test_make_bad_classification_schema():
    # Typo in costFunction
    schema = makeSchema(n_in = 100, 
                        n_out = 5, 
                        modelType = "classification", 
                        costFunction = "negative-loglikelihood")
    

def test_make_good_classification_schema():
    schema = makeSchema(n_in = 100, 
                        n_out = 5, 
                        modelType = "classification", 
                        costFunction = "negative-log-likelihood")

    validateSchema(schema)

