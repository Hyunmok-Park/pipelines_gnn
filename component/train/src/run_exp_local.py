import argparse
import sys
import torch

from inference_runner import *

torch.set_printoptions(profile='full')

def main(train_data, val_data):

    print("===========================")
    print("LOADING DATA")
    print("===========================")

    runner = NeuralInferenceRunner().train(train_data, val_data)
    sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data')
    parser.add_argument('--val_data')
    args = parser.parse_args()

    main(args.train_data, args.val_data)
