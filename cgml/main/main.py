
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

def resolveModel(args,log = None):

    if args.load:

        if log:
            log.write('\nLoading model from file: ' + args.load + '\n')
        
        model = ComputationalGraph.loadFromFile(args.load)
        
    elif args.cg:
        
        # Create the model
        model = ComputationalGraph(schema = yaml.load(open(args.cg,'r')),
                                   log = log,
                                   supCostWeight = args.supCostWeight,
                                   unsupCostWeight = args.unsupCostWeight,
                                   seed = args.seed,
                                   epsilon = args.epsilon,
                                   decay = args.decay,
                                   momentum = args.momentum)
        
    if args.recompileOnLoad:
        model.compile(log = log)

    if log:
        log.write(str(model._predict.maker.fgraph.toposort())+'\n')

    # Check if GPU is being used or not
    if log and np.any([ isinstance(x.op, T.Elemwise) 
                        for x in model._predict.maker.fgraph.toposort()]):
        log.write('No GPU found -- using CPU instead\n')
    else:
        log.write('Found GPU -- using that whenever possible\n')


    # Write description of the model to log streamx
    if log:
        log.write('\n' + str(model) + '\n')

    return model

def resolveValidationData(args,targetType,log = None):

    if args.validData:
        
        if log:
            log.write("Caching validation data for monitoring\n")
            
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


def startTrainingRoutine(model,args,log = None):

    sampleID,xValid,yValid = resolveValidationData(args,model.targetType,log = log)
        
    if args.trainData or args.trainDataStream:
        
        if log:
            log.write("Starting to read input data for training\n")
        
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

            if log:
                log.write("Pass " + str(passIdx) + 
                          " with mini-batch size " + 
                          str(miniBatchSize) + "\n")
            
            trainLog = model.train(drTrain = drTrain,
                                   X_valid = xValid,
                                   y_valid = yValid,
                                   miniBatchSize = miniBatchSize,
                                   verbose = args.verbose,
                                   log = log)
            
            if log:
                log.write(str(trainLog) + '\n')
            
            if args.nPasses > 1:
                drTrain.rewind()

    if log:
        log.write('\n')

def startTestingRoutine(model,args,outStream,log = None):

    if log:
        log.write("Starting to read data for prediction\n")
    
    if not args.predictions and log:
        log.write("WARNING: no predictions file provided! Predictions will not be saved\n")

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


