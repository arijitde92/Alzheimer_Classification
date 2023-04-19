import argparse
from train import train_main
from test import test_main

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-b', '--batch',
    type=int,
    dest='batch_size',
    default=32,
    help='Batch size denoting the number of images loaded into the model in each iteration'
)
parser.add_argument(
    '-e', '--epochs',
    type=int,
    default=50,
    help='Number of epochs to train our network for'
)
parser.add_argument(
    '-lr', '--learning-rate',
    type=float,
    dest='learning_rate',
    default=0.0000027,
    help='Learning rate for training the model'
)
parser.add_argument(
    '-v', '--valsplit',
    type=float,
    default=0.2,
    help='Validation Split ratio',
)
parser.add_argument(
    '-m', '--model',
    type=str,
    default="Trained_Models",
    help='Path to trained model',
)
args = vars(parser.parse_args())

TRAIN_ROOT_PATH = 'Data/Train'
TEST_ROOT_PATH = "Data/Test"
BATCH_SIZE = args['batch_size']
N_EPOCHS = args['epochs']
LR = args['learning_rate']
VAL_SPLIT = args['valsplit']
class_mapping = {"CN": 0, "EMCI": 1, "LMCI": 2, "AD": 3}

if __name__ == '__main__':
    train_main(TRAIN_ROOT_PATH, N_EPOCHS, BATCH_SIZE, VAL_SPLIT, LR, class_mapping)
    test_main(TEST_ROOT_PATH, args['model'], BATCH_SIZE, class_mapping)
