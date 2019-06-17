"""Predict labels using a second-level classifier.

This script trains a neural network classifier on the training set
meta-features created using the ``meta_features.py`` script.

This script requires three command-line arguments:

  * train_path: Path to training features.
  * metadata_path: Path to training metadata.
  * output_path: Output folder path.

It also takes optional arguments:

  * --n_folds: Number of folds in the training set. Ignored if
    predictions are to be generated for the test set. Default: 4.
  * --test_path: Path to test features. If this is specified, the script
    will generate predictions for the test set and write them to a
    submission file. Otherwise, it will generate predictions for the
    training set on a fold-by-fold basis and write them to a csv file.
"""

import argparse

import h5py
import numpy as np
import pandas as pd

import training
import utils


BLOCK_SIZE = 100


def train_and_predict(x_train, y_train, x_val, y_val, x_test):
    # Reshape data: [M * N, K] -> [M, N, K]
    n_feats = x_train.shape[1]
    n_classes = y_train.shape[1]
    x_train = np.reshape(x_train, (-1, BLOCK_SIZE, n_feats))
    x_val = np.reshape(x_val, (-1, BLOCK_SIZE, n_feats))
    x_test = np.reshape(x_test, (-1, BLOCK_SIZE, n_feats))
    y_train = np.reshape(y_train, (-1, BLOCK_SIZE, n_classes))
    y_val = np.reshape(y_val, (-1, BLOCK_SIZE, n_classes))

    return training.train_and_predict(x_train, y_train, x_val, y_val, x_test)


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('train_path', help='path to training features')
parser.add_argument('metadata_path', help='path to training metadata')
parser.add_argument('output_path', help='output file path')
parser.add_argument('--n_folds', type=int, default=4, help='number of folds')
parser.add_argument('--test_path', help='path to test features')
args = parser.parse_args()

# Load training data
with h5py.File(args.train_path, 'r') as f:
    x_train = np.array(f['F'])

df_train = utils.read_metadata(args.metadata_path, BLOCK_SIZE)
y_train = pd.get_dummies(df_train).values

if args.test_path:
    # Load test data
    with h5py.File(args.test_path, 'r') as f:
        x_test = np.array(f['F'])
        index = pd.Index(f['names'])

    # Use a subset of the training data for validation
    cutoff = len(index) * 0.15
    y_pred = train_and_predict(x_train[cutoff:],
                               y_train[cutoff:],
                               x_train[:cutoff],
                               y_train[:cutoff],
                               x_test)
else:
    # Train and predict for each fold and concatenate the predictions
    y_preds = []
    index = pd.Index([])
    for fold in range(1, args.n_folds + 1):
        mask = df_train.index.str.startswith('split%d' % fold)
        index = index.append(df_train[mask].index.str.replace('gt', 'prob'))

        y_pred = train_and_predict(x_train[~mask],
                                   y_train[~mask],
                                   x_train[mask],
                                   y_train[mask],
                                   x_train[mask])
        y_preds.append(y_pred)

        f1_score = utils.f1_score(y_train[mask], y_pred)
        print('[Fold %d] F1 score: %f\n' % (fold, f1_score))

    y_pred = np.concatenate(y_preds)

    # Compute predictions based on mean ensembling
    y_avg = x_train.reshape(x_train.shape[:-1] + (-1, 11)).mean(axis=1)

    # Print performance results
    print('Mean F1 score: %f' % utils.f1_score(y_train, y_avg))
    print('Stacking F1 score: %f' % utils.f1_score(y_train, y_pred))

# Write to CSV files
df_pred = pd.DataFrame(y_pred, index=index)
utils.write_metadata(df_pred, args.metadata_path, args.output_path)
