import os

import numpy as np
import pandas as pd
import sklearn.metrics as metrics


def trim_clips(model_dirs):
    dfs = []
    for fn in os.sorted(os.listdir(model_dirs[0])):
        df = pd.read_csv(os.path.join(model_dirs[0], fn), index_col=0).values
        dfs.append(df)
        fn_len = df.shape[0]
        for model_dir in model_dirs[1:]:
            df = pd.read_csv(os.path.join(model_dir, fn), index_col=0).values
            dfs.append(df)
            fn_len = min(df.shape[0], fn_len)
        for idx, model_dir in enumerate(model_dirs):
            dfs[idx].to_csv(os.path.join(model_dir, fn))


def read_metadata(path, block_size=100):
    return pd.concat([read_metadata_file(path, fname, block_size) 
        for fname in sorted(os.listdir(path))])


def read_metadata_file(path, fname, block_size=100):
    df = pd.read_csv(os.path.join(path, fname), index_col=0)

    # Pad so that number of frames is multiple of block_size
    padding_size = block_size - len(df) % block_size
    if block_size > 0 and padding_size < block_size:
        padding = pd.DataFrame(np.zeros_like(df[:padding_size]),
                               columns=df.columns)
        df = pd.concat([df, padding], ignore_index=True, copy=False)

    # Prepend name of file to indexes
    df.index = ['%s_%d' % (fname, idx) for idx in df.index]

    return df


def write_metadata(df_all, ref_path, output_path):
    for fname in sorted(os.listdir(ref_path)):
        df = read_metadata_file(ref_path, fname, block_size=-1)
        index = df.index.str.replace('gt', 'prob')
        df_pred = df_all.loc[index].reset_index(drop=True)
        df_pred.to_csv(os.path.join(output_path, fname))


def f1_score(y_true, y_pred, threshold=0.5):
    return metrics.f1_score(y_true, y_pred > threshold, average='micro')
