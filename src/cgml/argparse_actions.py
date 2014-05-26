
import argparse,random

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
        '--learnRate',
        help = 'Learning rate for the stochastic gradient descent algorithm',
        type = float,
        dest = 'learnRate',
        default = 0.01,
        required = False)

    parser.add_argument(
        '--momentum',
        help = 'How much previous gradient update affects the current one',
        type = float,
        dest = 'momentum',
        default = 0.0,
        required = False)

    parser.add_argument(
        '--L1Reg',
        help = 'L1 Regularization term',
        type = float,
        dest = 'L1Reg',
        default = 0.0,
        required = False)

    parser.add_argument(
        '--L2Reg',
        help = 'L2 regularization term',
        type = float,
        dest = 'L2Reg',
        default = 0.0,
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
        '--nPasses',
        help = 'How many passes through the data we go',
        type = int,
        default = 1,
        required = False)

    parser.add_argument(
        '--batchSize',
        help = 'Number of samples per mini-batch',
        type = int,
        dest = 'batchSize',
        default = 1,
        required = False)
    
    parser.add_argument(
        '--log',
        help = 'Provide log file',
        type = str,
        default = None,
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

    args = parser.parse_args()

    if log:

        log.write('\nParsed the following arguments:\n')
        log.write(' --verbose         ' + str(args.verbose)         + '\n')
        log.write(' --seed            ' + str(args.seed)            + '\n')
        log.write(' --cg              ' + str(args.cg)              + '\n')
        log.write(' --trainData       ' + str(args.trainData)       + '\n')
        log.write(' --validData       ' + str(args.validData)       + '\n')
        log.write(' --testData        ' + str(args.testData)        + '\n')
        log.write(' --learnRate       ' + str(args.learnRate)       + '\n')
        log.write(' --momentum        ' + str(args.momentum)        + '\n')
        log.write(' --L1Reg           ' + str(args.L1Reg)           + '\n')
        log.write(' --L2Reg           ' + str(args.L2Reg)           + '\n')
        log.write(' --supCostWeight   ' + str(args.supCostWeight)   + '\n')
        log.write(' --unsupCostWeight ' + str(args.unsupCostWeight) + '\n')
        log.write(' --batchSize       ' + str(args.batchSize)       + '\n')
        log.write(' --nPasses         ' + str(args.nPasses)         + '\n')
        log.write(' --log             ' + str(args.log)             + '\n')
        log.write(' --save            ' + str(args.save)            + '\n')
        log.write(' --load            ' + str(args.load)            + '\n')

    if not args.cg and not args.load:
        raise Exception("You must provide either a Computational Graph schema file (--cg) " + 
                        "or model to load (--load)")
        
    if args.cg and args.load:
        raise Exception("Cannot provide both Computational Graph schema file (--cg) " + 
                        "and model to load (--load)")

    if args.cg and not args.trainData:
        raise Exception("Computational Graph schema file provided, " + 
                        "but no input data for training (--trainData)")

    if args.load and not args.trainData and not args.testData:
        raise Exception("Model file to load provided, " + 
                        "but no data for training (--trainData) or prediction (--testData)")

    return args










