# Deep learning course

import os, argparse

def get_parser(parser):

    #parser = argparse.ArgumentParser()

    '''
    Model Parameters
    '''
    parser.add_argument('--rnn_type', type = str, default = 'LSTM')

    parser.add_argument('--hidden_size', type=int, default=64)

    parser.add_argument('--lstm_layers', type=int, default=2)

    '''
    Training Parameters
    '''
    parser.add_argument('--num_epochs', type=int, default=3500, help='maximum number of epochs')

    parser.add_argument('--batch_size', type=int, default=100)

    parser.add_argument('--learning_rate', type=float, default=.1, help='base learning rate')

    parser.add_argument('--clip', type=bool, default=False)

    parser.add_argument('--dropout', type=float, default=0, help='dropout ratio')

    parser.add_argument('--data_split', type=float, default=0.15, help = "percentage of validation data")

    parser.add_argument('--data_dir', type=str, default='./data/')

    parser.add_argument('--npy_file', type=str, default='Robot_Trials_DL.npy')
                                                                                                                        
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--mode', type=str, default='train')

    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--comment', type=str, default = 'test', help='name for tensorboardX')

    args = parser.parse_args()

    return args
