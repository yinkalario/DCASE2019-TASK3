"""Generate meta-features for stacking. Creted by Turab Iqbal.

Given a number of trained models, the user can use this script to
generate features based on the predictions of these models. For example,
if we have five models, and each model outputs an N x K matrix of
predictions, where N is the number of predicted audio clips and K=11 is
the number of classes, this script concatenates these to produce an N x
5K matrix, i.e. N feature vectors.

This script requires two command-line arguments:

  * pred_path: Path to predictions directory.
  * output_path: Output file path of meta-features.
"""

import argparse
import os.path

import h5py
import numpy as np
import pandas as pd

import utils


# Relative paths of the model predictions
MODELS = ['model_1', 'model_2', 'model_3', 'model_4']


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('pred_path', help='path to predictions directory')
parser.add_argument('output_path', help='output file path')
args = parser.parse_args()

# Trim the time length of the input files
model_dirs = []
for model in MODELS:
    model_dirs.append(os.path.join(args.pred_path, model))
utils.trim_clips(model_dirs)

# Collect predictions for each model
preds = []
feats = []
for model in MODELS:
    df = utils.read_metadata(os.path.join(args.pred_path, model))
    preds.append((df > 0.5).astype(int).unstack())
    feats.append(df)

# Print correlation matrix
print(pd.concat(preds, axis=1).corr())

# Save meta-features to disk
feats = np.stack(feats, axis=1)
feats = np.reshape(feats, (feats.shape[0], -1))
with h5py.File(args.output_path, 'w') as f:
    f.create_dataset('F', data=feats)
    f.create_dataset('names', data=preds[0].index.levels[1],
                     dtype=h5py.special_dtype(vlen=str))
