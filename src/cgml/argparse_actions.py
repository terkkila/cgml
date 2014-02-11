
import argparse

def giveArgs(log = None):

    parser = argparse.ArgumentParser( description = 'Machine Learning with Computational Graphs.')
    
    parser.add_argument(
        '--cg',
        help     = 'Computational Graph schema file',
        dest     = 'cg',
        default  = None,
        required = True)
    
    parser.add_argument(
        '--trainData',
        help     = 'Data for training',
        dest     = 'trainData',
        default  = None,
        required = True)

    parser.add_argument(
        '--testData',
        help     = 'Data for testing',
        dest     = 'testData',
        default  = None,
        required = True)

    parser.add_argument(
        '--learnRate',
        help = 'Learning rate for the stochastic gradient descent algorithm',
        type = float,
        dest = 'learnRate',
        default = 0.01,
        required = False)

    parser.add_argument(
        '--nPasses',
        help = 'How many passes through the data we go',
        type = int,
        default = 1,
        required = False)

    parser.add_argument(
        '--log',
        help = 'Provide log file',
        type = str,
        default = None,
        required = False)
    
    args = parser.parse_args()

    if log:

        log.write('\nParsed the following arguments:\n')
        log.write(' --cg        ' + str(args.cg)         + '\n')
        log.write(' --trainData ' + args.trainData       + '\n')
        log.write(' --testData  ' + args.testData        + '\n')
        log.write(' --learnRate ' + str(args.learnRate)  + '\n')
        log.write(' --nPasses   ' + str(args.nPasses)    + '\n')
        log.write(' --log       ' + args.log             + '\n')
        
    return args










