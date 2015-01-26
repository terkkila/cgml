
import argparse,random

from cgml.constants import DEFAULT_ADADELTA_EPSILON
from cgml.constants import DEFAULT_ADADELTA_DECAY
from cgml.constants import DEFAULT_ADADELTA_MOMENTUM
from cgml.constants import DEFAULT_MINI_BATCH_SIZE
from cgml.constants import DEFAULT_MINI_BATCH_INCREMENT
from cgml.constants import DEFAULT_DEVICE_BATCH_SIZE

def giveArgs(log = None):

    parser = argparse.ArgumentParser( description = 'Machine Learning with Computational Graphs.')

    parser.add_argument(
        '--verbose',
        help     = 'Print extra information?',
        dest     = 'verbose',
        default  = False,
        action = 'store_true',
        required = False)

    
    parser.add_argument(
        '--seed',
        help     = 'Random Number seed',
        type = int,
        dest     = 'seed',
        default  = None,
        required = False)
    

    parser.add_argument(
        '--cg',
        help     = 'Computational Graph schema file',
        type = str,
        dest     = 'cg',
        default  = None,
        required = False)
    
    parser.add_argument(
        '--trainData',
        help     = 'Data for training',
        type = str,
        dest     = 'trainData',
        default  = None,
        required = False)
    
    parser.add_argument(
        '--trainDataStream',
        help     = 'Streaming data for training',
        dest     = 'trainDataStream',
        default  = False,
        action = 'store_true',
        required = False)

    parser.add_argument(
        '--validData',
        help     = 'Data for validation',
        type = str,
        dest     = 'validData',
        default  = None,
        required = False)
    
    parser.add_argument(
        '--testData',
        help     = 'Data for testing',
        type = str,
        dest     = 'testData',
        default  = None,
        required = False)

    parser.add_argument(
        '--supCostWeight',
        help = 'How much weight is given to supervised cost',
        type = float,
        dest = 'supCostWeight',
        default = 1.0,
        required = False)

    parser.add_argument(
        '--unsupCostWeight',
        help = 'How much weight is given to unsupervised cost',
        type = float,
        dest = 'unsupCostWeight',
        default = 1.0,
        required = False)

    parser.add_argument(
        '--deviceBatchSize',
        help = 'Number of samples stored in device',
        type = int,
        dest = 'deviceBatchSize',
        default = DEFAULT_DEVICE_BATCH_SIZE,
        required = False)
    
    parser.add_argument(
        '--miniBatchSize',
        help = 'Number of samples used in device to update parameters',
        type = int,
        dest = 'miniBatchSize',
        default = DEFAULT_MINI_BATCH_SIZE,
        required = False)
    

    parser.add_argument(
        '--miniBatchIncrement',
        help = 'Number of samples to increment the mini batch size with',
        type = int,
        dest = 'miniBatchIncrement',
        default = DEFAULT_MINI_BATCH_INCREMENT,
        required = False)
    

    parser.add_argument(
        '--save',
        help = 'Save model to file',
        type = str,
        default = None,
        required = False)

    parser.add_argument(
        '--load',
        help = 'Load model from file',
        type = str,
        default = None,
        required = False)

    parser.add_argument(
        '--recompileOnLoad',
        help = 'If the architecture gets changed, one can enforce recompilation',
        action = 'store_true',
        default = False,
        required = False)

    parser.add_argument(
        "--predictions",
        help = "Print predictions to file",
        default = None,
        required = False)

    parser.add_argument(
        '--getImportance',
        help = 'Append importance scores to predictions',
        default = False,
        action = 'store_true',
        required = False)

    parser.add_argument(
        "--epsilon",
        help = "Epsilon term for AdaDelta",
        type = float,
        default = DEFAULT_ADADELTA_EPSILON,
        required = False)

    parser.add_argument(
        "--decay",
        help = "Decay term for AdaDelta",
        type = float,
        default = DEFAULT_ADADELTA_DECAY,
        required = False)

    parser.add_argument(
        "--momentum",
        help = "Momentum term for AdaDelta",
        type = float,
        default = DEFAULT_ADADELTA_MOMENTUM,
        required = False)

    parser.add_argument(
        '--ensemble',
        help = 'If greater than 1, ensemble predictor will be made',
        default = 1,
        type = int,
        required = False)

    parser.add_argument(
        '--nPasses',
        help = 'How many passes over the data',
        default = 1,
        type = int,
        required = False)

    args = parser.parse_args()

    if log:

        log.write('\nParsed the following arguments:\n')
        log.write(' --verbose            ' + str(args.verbose)            + '\n')
        log.write(' --seed               ' + str(args.seed)               + '\n')
        log.write(' --cg                 ' + str(args.cg)                 + '\n')
        log.write(' --trainData          ' + str(args.trainData)          + '\n')
        log.write(' --trainDataStream    ' + str(args.trainDataStream)    + '\n')
        log.write(' --validData          ' + str(args.validData)          + '\n')
        log.write(' --testData           ' + str(args.testData)           + '\n')
        log.write(' --predictions        ' + str(args.predictions)        + '\n')
        log.write(' --getImportance      ' + str(args.getImportance)      + '\n')
        log.write(' --supCostWeight      ' + str(args.supCostWeight)      + '\n')
        log.write(' --unsupCostWeight    ' + str(args.unsupCostWeight)    + '\n')
        log.write(' --deviceBatchSize    ' + str(args.deviceBatchSize)    + '\n')
        log.write(' --miniBatchSize      ' + str(args.miniBatchSize)      + '\n')
        log.write(' --miniBatchIncrement ' + str(args.miniBatchIncrement) + '\n')
        log.write(' --nPasses            ' + str(args.nPasses)            + '\n')
        log.write(' --epsilon            ' + str(args.epsilon)            + '\n')
        log.write(' --decay              ' + str(args.decay)              + '\n')
        log.write(' --momentum           ' + str(args.momentum)           + '\n')
        log.write(' --save               ' + str(args.save)               + '\n')
        log.write(' --load               ' + str(args.load)               + '\n')
        log.write(' --ensemble           ' + str(args.ensemble)           + '\n')
        log.write(' --recompileOnLoad    ' + str(args.recompileOnLoad)    + '\n')

    if not args.cg and not args.load:
        raise Exception("You must provide either a Computational Graph schema file (--cg) " + 
                        "or model to load (--load)")
        
    if args.cg and args.load:
        raise Exception("Cannot provide both Computational Graph schema file (--cg) " + 
                        "and model to load (--load)")

    if args.load and not (args.trainData or args.trainDataStream) and not args.testData:
        raise Exception("Model file to load provided, " + 
                        "but no data for training (--trainData) or prediction (--testData)")

    if args.miniBatchSize >= args.deviceBatchSize:
        raise Exception("--minibatchSize should not be larger than --deviceBatchSize")

    if args.ensemble < 1:
        raise Exception("--ensemble cannot be smaller than 1")

    return args










