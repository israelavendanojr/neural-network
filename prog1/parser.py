import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-train_feat', type=str, required=True,
                        help='Training set feature file')
    parser.add_argument('-train_target', type=str, required=True,
                        help='Training set target file')
    parser.add_argument('-dev_feat', type=str, required=True,
                        help='Development set feature file')
    parser.add_argument('-dev_target', type=str, required=True,
                        help='Development set target file')
    parser.add_argument('-nunits', type=int, required=True,
                        help='Number of hidden units per layer')
    parser.add_argument('-nlayers', type=int, required=True,
                        help='Number of hidden layers')
    parser.add_argument('-hidden_act', type=str, required=True, choices=['sig', 'tanh', 'relu'],
                        help='Hidden unit activation function')
    parser.add_argument('-type', type=str, required=True, choices=['C', 'R'],
                        help='Problem mode: C for classification, R for regression')
    parser.add_argument('-output_dim', type=int, required=True,
                        help='Number of classes or output dimension')
    parser.add_argument('-total_updates', type=int, required=True,
                        help='Total number of updates (gradient steps)')
    parser.add_argument('-learnrate', type=float, required=True,
                        help='Learning rate')
    parser.add_argument('-init_range', type=float, required=True,
                        help='Range for uniform random initialization')
    parser.add_argument('-mb', type=int, required=True,
                        help='Minibatch size (0 for full batch)')
    parser.add_argument('-report_freq', type=int, required=True,
                        help='Frequency of reporting performance')
    parser.add_argument('-v', action='store_true',
                        help='Verbose mode')
    
    return parser.parse_args()