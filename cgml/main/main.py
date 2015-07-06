
import yaml
import json
import sys
import numpy as np
import math
import theano.tensor as T

from cgml.graph import ComputationalGraph
from cgml.io import ppf
from cgml.io import DataReader


def resolveOutStream(args):

    # Print outputs here
    if args.predictions:
        outStream = open(args.predictions,'w')
    else:
        outStream = None

    return outStream

def resolveModel(args,logger):

    if args.load:

        if log:
            log.write('\nLoading model from file: ' + args.load + '\n')
        
        model = ComputationalGraph.loadFromFile(args.load)
        
    elif args.cg:
        
        # Create the model
        model = ComputationalGraph(schema = yaml.load(open(args.cg,'r')),
                                   logger = logger,
                                   supCostWeight = args.supCostWeight,
                                   unsupCostWeight = args.unsupCostWeight,
                                   seed = args.seed,
                                   epsilon = args.epsilon,
                                   decay = args.decay,
                                   momentum = args.momentum)
        
    if args.recompileOnLoad:
        model.compile(logger = logger)

    logger.info(str(model._predict.maker.fgraph.toposort()))

    # Check if GPU is being used or not
    if np.any([ isinstance(x.op, T.Elemwise) 
                for x in model._predict.maker.fgraph.toposort()]):
        logger.info('No GPU found -- using CPU instead\n')
    else:
        logger.info('Found GPU -- using that whenever possible\n')


    # Write description of the model to log streamx
    logger.info('\n' + str(model) + '\n')

    return model

def resolveValidationData(args,targetType,log = None):

    if args.validData:
        
        logger.info("Caching validation data for monitoring")
            
        sampleIDs,xValid,yValid = DataReader(args.validData,
                                             targetType = targetType).cache()
        
    else:
        
        sampleIDs,xValid,yValid = None,None,None

    return sampleIDs,xValid,yValid


def strVec(vec):
    if type(vec) in [np.int64,int,np.float64,float]:
        return str(vec)
    return ' '.join(map(str,vec))


def strMat(mat):
    return '\n'.join(strVec(map(strVec,row)) for row in mat)


def startTrainingRoutine(model,args,logger):

    sampleID,xValid,yValid = resolveValidationData(args,model.targetType,logger)
        
    if args.trainData or args.trainDataStream:
        
        logger.info("Starting to read input data for training")
        
        if args.trainData:
            
            # Data reader from file
            drTrain = DataReader(args.trainData,
                                 batchSize = args.deviceBatchSize,
                                 targetType = model.targetType)
            
        else:
            
            # Data reader from stream
            drTrain = DataReader(sys.stdin,
                                 batchSize = args.deviceBatchSize,
                                 targetType = model.targetType)
            
        for passIdx in xrange(args.nPasses):
            
            # Incrementing mini batch size if increment is > 0
            miniBatchSize = args.miniBatchSize + passIdx * args.miniBatchIncrement

            logger.info("Pass " + str(passIdx) + 
                        " with mini-batch size " + 
                        str(miniBatchSize))
            
            trainLog = model.train(drTrain = drTrain,
                                   X_valid = xValid,
                                   y_valid = yValid,
                                   miniBatchSize = miniBatchSize,
                                   verbose = args.verbose)
            
            logger.info(str(trainLog))
            
            if args.nPasses > 1:
                drTrain.rewind()

def startTestingRoutine(model,args,outStream,logger):

    logger.info("Starting to read data for prediction\n")
    
    if not args.predictions:
        logger.info("WARNING: no predictions file provided! Predictions will not be saved\n")

    # Data reader for prediction
    drTest  = DataReader(args.testData,
                         batchSize = 1,
                         targetType = model.targetType)
    
    # Start prediction
    for sampleIDs,x_test,y_test in drTest:

        if model.type in ['classification','regression']:
            
            for sampleID,xi_test,yi_test,yhati_test,impi_test in zip(sampleIDs,
                                                                     x_test,
                                                                     y_test,
                                                                     model.predict(x_test),
                                                                     model.importance(x_test)):
                
                predObj = {"sampleID":sampleID,
                           "y":yi_test.tolist(),
                           "yhat":yhati_test.tolist()}

                if args.getImportance:
                    predObj["importance"] = impi_test
                
                if outStream:
                    outStream.write(json.dumps(predObj) + "\n")
                            
        elif model.type == 'autoencoder':
            
            if outStream:
                outStream.write( strMat(zip([y_test],model.encode(x_test))) + '\n' )
            
        elif model.type == 'supervised-autoencoder':
            
            if outStream:
                outStream.write( strMat(zip([y_test],
                                            model.predict(x_test),
                                            model.encode(x_test),
                                            model.decode(x_test))) + '\n' )


