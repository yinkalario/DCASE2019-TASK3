import pdb
import random

import numpy as np
import torch


def mixup(batch_x, batch_y, alpha):
    '''
    Mixup for data augmentation

    Input:
        batch_x: (batch_size, channel_num, time, frequency)
        batch_y: (batch_size, time, class_num)
        alpha: 0.0-1.0, 0.5
        cuda: using gpu
    '''
    device = batch_x.device

    batch_size = batch_x.shape[0]

    indexes = np.arange(batch_size)
    np.random.shuffle(indexes)
    lams = torch.Tensor(np.random.beta(alpha, alpha, batch_size)).to(device)

    mixed_x = lams[:,None,None,None] * batch_x + (1. - lams[:,None,None,None]) * batch_x[indexes]
    mixed_y = lams[:,None,None] * batch_y + (1. - lams[:,None,None]) * batch_y[indexes]

    return mixed_x, mixed_y


def freq_mask(spec, ratio_F=0.1, num_masks=1, replace_with_zero=False):
    '''
    Frequency mask from SpecAug

    Input:
        spec: input spectrograms
        ratio_F: mask ratio for frequency bins
        num_masks: number of masks
        replace_with_zero: False (default) for replace masked frequency components with mean, True for replace masked frequency components with zero.
    '''
    cloned = spec.clone()
    batch_size, _, _, frequency_bins = cloned.shape
    F = int(ratio_F * frequency_bins)
    
    for batch_idx in range(batch_size):
        for _ in range(0, num_masks):        
            f = random.randrange(0, F)
            f_zero = random.randrange(0, frequency_bins - f)

            # avoids randrange error if values are equal and range is empty
            if (f_zero == f_zero + f): 
                return cloned
            mask_end = random.randrange(f_zero, f_zero + f)

            if (replace_with_zero): 
                cloned[batch_idx, :, :, f_zero:mask_end] = 0
            else: 
                cloned[batch_idx, :, :, f_zero:mask_end] = cloned.mean()
    return cloned


def time_mask(spec, ratio_T=0.1, num_masks=1, replace_with_zero=False):
    '''
    Temporal mask from SpecAug

    Input:
        spec: input spectrograms
        ratio_T: mask ratio for temporal bins
        num_masks: number of masks
        replace_with_zero: False (default) for replace masked temporal components with mean, True for replace masked temporal components with zero.
    '''
    cloned = spec.clone()
    batch_size, _, time_bins, _ = cloned.shape
    T = int(ratio_T * time_bins)

    for batch_idx in range(batch_size):
        for _ in range(0, num_masks):
            t = random.randrange(0, T)
            t_zero = random.randrange(0, time_bins - t)

            # avoids randrange error if values are equal and range is empty
            if (t_zero == t_zero + t): 
                return cloned
            mask_end = random.randrange(t_zero, t_zero + t)

            if (replace_with_zero): 
                cloned[batch_idx, :, t_zero:mask_end, :] = 0
            else: 
                cloned[batch_idx, :, t_zero:mask_end, :] = cloned.mean()
    return cloned
