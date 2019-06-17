import argparse
import os
import pdb
import sys
from timeit import default_timer as timer

import h5py
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal
from tqdm import tqdm

from utilities import calculate_scalar, event_labels, lb_to_ix

fs = 32000
nfft = 1024
hopsize = 320 # 640 for 20 ms
mel_bins = 128
window = 'hann'
fmin = 50
hdf5_folder_name = '{}fs_{}nfft_{}hs_{}melb'.format(fs, nfft, hopsize, mel_bins)


class LogMelExtractor():
    def __init__(self, fs, nfft, hopsize, mel_bins, window, fmin):

        self.nfft = nfft
        self.hopsize = hopsize
        self.window = window
        self.melW = librosa.filters.mel(sr=fs,
                                        n_fft=nfft,
                                        n_mels=mel_bins,
                                        fmin=fmin)

    def transform(self, audio):

        channel_num = audio.shape[0]
        feature_logmel = []

        for n in range(channel_num):
            S = np.abs(librosa.stft(y=audio[n],
                                    n_fft=self.nfft,
                                    hop_length=self.hopsize,
                                    center=True,
                                    window=self.window,
                                    pad_mode='reflect'))**2

            S_mel = np.dot(self.melW, S).T
            S_logmel = librosa.power_to_db(S_mel, ref=1.0, amin=1e-10, top_db=None)
            S_logmel = np.expand_dims(S_logmel, axis=0)
            feature_logmel.append(S_logmel)

        feature_logmel = np.concatenate(feature_logmel, axis=0)

        return feature_logmel


class LogMelGccExtractor():
    def __init__(self, fs, nfft, hopsize, mel_bins, window, fmin):

        self.nfft = nfft
        self.hopsize = hopsize
        self.window = window
        self.melW = librosa.filters.mel(sr=fs,
                                        n_fft=nfft,
                                        n_mels=mel_bins,
                                        fmin=fmin)

    def logmel(self, sig):

        S = np.abs(librosa.stft(y=sig,
                                n_fft=self.nfft,
                                hop_length=self.hopsize,
                                center=True,
                                window=self.window,
                                pad_mode='reflect'))**2        
        S_mel = np.dot(self.melW, S).T
        S_logmel = librosa.power_to_db(S_mel, ref=1.0, amin=1e-10, top_db=None)
        S_logmel = np.expand_dims(S_logmel, axis=0)

        return S_logmel

    def gcc_phat(self, sig, refsig):

        ncorr = 2*self.nfft - 1
        nfft = int(2**np.ceil(np.log2(np.abs(ncorr))))
        Px = librosa.stft(y=sig,
                        n_fft=nfft,
                        hop_length=self.hopsize,
                        center=True,
                        window=self.window, 
                        pad_mode='reflect')
        Px_ref = librosa.stft(y=refsig,
                            n_fft=nfft,
                            hop_length=self.hopsize,
                            center=True,
                            window=self.window,
                            pad_mode='reflect')
    
        R = Px*np.conj(Px_ref)

        n_frames = R.shape[1]
        gcc_phat = []
        for i in range(n_frames):
            spec = R[:, i].flatten()
            cc = np.fft.irfft(np.exp(1.j*np.angle(spec)))
            cc = np.concatenate((cc[-mel_bins//2:], cc[:mel_bins//2]))
            gcc_phat.append(cc)
        gcc_phat = np.array(gcc_phat)
        gcc_phat = gcc_phat[None,:,:]

        return gcc_phat

    def transform(self, audio):

        channel_num = audio.shape[0]
        feature_logmel = []
        feature_gcc_phat = []
        for n in range(channel_num):
            feature_logmel.append(self.logmel(audio[n]))
            for m in range(n+1, channel_num):
                feature_gcc_phat.append(
                    self.gcc_phat(sig=audio[m], refsig=audio[n]))
        
        feature_logmel = np.concatenate(feature_logmel, axis=0)
        feature_gcc_phat = np.concatenate(feature_gcc_phat, axis=0)
        feature = np.concatenate([feature_logmel, feature_gcc_phat])

        return feature


class LogMelIntensityExtractor():
    def __init__(self, fs, nfft, hopsize, mel_bins, window, fmin):

        self.nfft = nfft
        self.hopsize = hopsize
        self.window = window
        self.melW = librosa.filters.mel(sr=fs,
                                        n_fft=nfft,
                                        n_mels=mel_bins,
                                        fmin=fmin)

    def logmel(self, sig):

        S = np.abs(librosa.stft(y=sig,
                                n_fft=nfft,
                                hop_length=self.hopsize,
                                center=True,
                                window=self.window,
                                pad_mode='reflect'))**2        
        S_mel = np.dot(self.melW, S).T
        S_logmel = librosa.power_to_db(S_mel, ref=1.0, amin=1e-10, top_db=None)
        S_logmel = np.expand_dims(S_logmel, axis=0)

        return S_logmel

    def intensity(self, sig):

        ref = sig[0]
        x = sig[1]
        y = sig[2]
        z = sig[3]

        Pref = librosa.stft(y=ref,
                        n_fft=nfft,
                        hop_length=hopsize,
                        center=True,
                        window=self.window, 
                        pad_mode='reflect')
        Px = librosa.stft(y=x,
                        n_fft=nfft,
                        hop_length=hopsize,
                        center=True,
                        window=self.window, 
                        pad_mode='reflect')
        Py = librosa.stft(y=y,
                        n_fft=nfft,
                        hop_length=hopsize,
                        center=True,
                        window=self.window, 
                        pad_mode='reflect')
        Pz = librosa.stft(y=z,
                        n_fft=nfft,
                        hop_length=hopsize,
                        center=True,
                        window=self.window, 
                        pad_mode='reflect')

        I1 = np.real(np.conj(Pref) * Px)
        I2 = np.real(np.conj(Pref) * Py)
        I3 = np.real(np.conj(Pref) * Pz)
        normal = np.sqrt(I1**2 + I2**2 + I3**2)
        I1 = np.dot(self.melW, I1 / normal).T
        I2 = np.dot(self.melW, I2 / normal).T
        I3 = np.dot(self.melW, I3 / normal).T
        intensity = np.array([I1, I2, I3])

        return intensity

    def transform(self, audio):

        channel_num = audio.shape[0]
        feature_logmel = []
        for n in range(0, channel_num):
            feature_logmel.append(self.logmel(audio[n]))
        feature_intensity = self.intensity(sig=audio)

        feature_logmel = np.concatenate(feature_logmel, axis=0)
        feature = np.concatenate([feature_logmel, feature_intensity], axis=0)

        return feature


class LogMelGccIntensityExtractor():
    def __init__(self, fs, nfft, hopsize, mel_bins, window, fmin):

        self.nfft = nfft
        self.hopsize = hopsize
        self.window = window
        self.melW = librosa.filters.mel(sr=fs,
                                        n_fft=nfft,
                                        n_mels=mel_bins,
                                        fmin=fmin)

    def logmel(self, sig):

        S = np.abs(librosa.stft(y=sig,
                                n_fft=self.nfft,
                                hop_length=self.hopsize,
                                center=True,
                                window=self.window,
                                pad_mode='reflect'))**2        
        S_mel = np.dot(self.melW, S).T
        S_logmel = librosa.power_to_db(S_mel, ref=1.0, amin=1e-10, top_db=None)
        S_logmel = np.expand_dims(S_logmel, axis=0)

        return S_logmel

    def gcc_phat(self, sig, refsig):

        ncorr = 2*self.nfft - 1
        nfft = int(2**np.ceil(np.log2(np.abs(ncorr))))
        Px = librosa.stft(y=sig,
                        n_fft=nfft,
                        hop_length=self.hopsize,
                        center=True,
                        window=self.window, 
                        pad_mode='reflect')
        Px_ref = librosa.stft(y=refsig,
                            n_fft=nfft,
                            hop_length=self.hopsize,
                            center=True,
                            window=self.window,
                            pad_mode='reflect')
    
        R = Px*np.conj(Px_ref)

        n_frames = R.shape[1]
        gcc_phat = []
        for i in range(n_frames):
            spec = R[:, i].flatten()
            cc = np.fft.irfft(np.exp(1.j*np.angle(spec)))
            cc = np.concatenate((cc[-mel_bins//2:], cc[:mel_bins//2]))
            gcc_phat.append(cc)
        gcc_phat = np.array(gcc_phat)
        gcc_phat = gcc_phat[None,:,:]

        return gcc_phat

    def intensity(self, sig):

        ref = sig[0]
        x = sig[1]
        y = sig[2]
        z = sig[3]

        Pref = librosa.stft(y=ref,
                        n_fft=nfft,
                        hop_length=hopsize,
                        center=True,
                        window=self.window, 
                        pad_mode='reflect')
        Px = librosa.stft(y=x,
                        n_fft=nfft,
                        hop_length=hopsize,
                        center=True,
                        window=self.window, 
                        pad_mode='reflect')
        Py = librosa.stft(y=y,
                        n_fft=nfft,
                        hop_length=hopsize,
                        center=True,
                        window=self.window, 
                        pad_mode='reflect')
        Pz = librosa.stft(y=z,
                        n_fft=nfft,
                        hop_length=hopsize,
                        center=True,
                        window=self.window, 
                        pad_mode='reflect')

        I1 = np.real(np.conj(Pref) * Px)
        I2 = np.real(np.conj(Pref) * Py)
        I3 = np.real(np.conj(Pref) * Pz)
        normal = np.sqrt(I1**2 + I2**2 + I3**2)
        I1 = np.dot(self.melW, I1 / normal).T
        I2 = np.dot(self.melW, I2 / normal).T
        I3 = np.dot(self.melW, I3 / normal).T
        intensity = np.array([I1, I2, I3])

        return intensity

    def transform(self, audio):
        
        feature_logmel = []
        for n in range(0, 4):
            feature_logmel.append(self.logmel(audio[n]))
        feature_intensity = self.intensity(sig=audio[0:4])
        feature_logmel = np.concatenate(feature_logmel, axis=0)
        feature_foa = np.concatenate([feature_logmel, feature_intensity], axis=0)

        feature_logmel = []
        feature_gcc_phat = []
        for n in range(4, 8):
            feature_logmel.append(self.logmel(audio[n]))
            for m in range(n+1, 8):
                feature_gcc_phat.append(
                    self.gcc_phat(sig=audio[m], refsig=audio[n]))
        feature_logmel = np.concatenate(feature_logmel, axis=0)
        feature_gcc_phat = np.concatenate(feature_gcc_phat, axis=0)
        feature_mic = np.concatenate([feature_logmel, feature_gcc_phat], axis=0)

        feature = np.concatenate([feature_foa, feature_mic], axis=0)

        return feature


def RT_preprocessing(extractor, audio):

    '''This step needs to be considered'''
    # audio = audio / (np.max(np.abs(audio)) + np.finfo(np.float).eps)

    feature = extractor.transform(audio)
    '''(channels, seq_len, mel_bins)'''
    '''(channels, time, frequency)'''

    return feature

def extract_dev_features(args):
    """
    Write features and infos of audios to hdf5.

    Args:
        dataset_dir: dataset path
        feature_dir: feature path
        audio_type: 'foa' | 'mic' | 'foa&mic'
    """
    # extractor
    if args.feature_type == 'logmel':
        extractor = LogMelExtractor(fs=fs, 
                                    nfft=nfft,
                                    hopsize=hopsize,
                                    mel_bins=mel_bins,
                                    window=window,
                                    fmin=fmin)  
    elif args.feature_type == 'logmelgcc':
        extractor = LogMelGccExtractor(fs=fs, 
                                    nfft=nfft,
                                    hopsize=hopsize,
                                    mel_bins=mel_bins,
                                    window=window,
                                    fmin=fmin)
    elif args.feature_type == 'logmelintensity':
        extractor = LogMelIntensityExtractor(fs=fs, 
                                    nfft=nfft,
                                    hopsize=hopsize,
                                    mel_bins=mel_bins,
                                    window=window,
                                    fmin=fmin)    
    elif args.feature_type == 'logmelgccintensity':
        extractor = LogMelGccIntensityExtractor(fs=fs, 
                                    nfft=nfft,
                                    hopsize=hopsize,
                                    mel_bins=mel_bins,
                                    window=window,
                                    fmin=fmin)

    # Path
    if args.feature_type == 'logmelgccintensity':
        audio_dir = [os.path.join(args.dataset_dir, 'dev', 'foa_dev'), os.path.join(args.dataset_dir, 'dev', 'mic_dev')]
        hdf5_dir = os.path.join(args.feature_dir, args.feature_type,
                                hdf5_folder_name, 'foa&mic_dev')
        os.makedirs(hdf5_dir, exist_ok=True)
    else:
        audio_dir = [os.path.join(args.dataset_dir,  'dev', args.audio_type + '_dev')]
        hdf5_dir = os.path.join(args.feature_dir, args.feature_type,
                                hdf5_folder_name, args.audio_type + '_dev')
        os.makedirs(hdf5_dir, exist_ok=True)

    meta_dir = os.path.join(args.dataset_dir, 'dev', 'metadata_dev')

    begin_time = timer()
    audio_count = 0

    print('\n============> Start Extracting Features\n')
    
    iterator = tqdm(sorted(os.listdir(audio_dir[0])), total=len(os.listdir(audio_dir[0])), unit='it')

    for audio_fn in iterator:

        if audio_fn.endswith('.wav') and not audio_fn.startswith('.'):

            fn = audio_fn.split('.')[0]
            if args.feature_type == 'logmelgccintensity':
                audio_path = [os.path.join(audio_dir[0], audio_fn), os.path.join(audio_dir[1], audio_fn)]
                audio_foa, _ = librosa.load(audio_path[0], sr=fs, mono=False, dtype=np.float32)
                audio_mic, _ = librosa.load(audio_path[1], sr=fs, mono=False, dtype=np.float32)
                audio_len = min(audio_foa.shape[1], audio_mic.shape[1])
                audio = np.concatenate([audio_foa[:, :audio_len], audio_mic[:, :audio_len]], axis=0)
                '''(channel_nums, samples)'''
            else:
                audio_path = os.path.join(audio_dir[0], audio_fn)
                audio, _ = librosa.load(audio_path, sr=fs, mono=False, dtype=np.float32)
                '''(channel_nums, samples)'''

            audio_count += 1

            if np.sum(np.abs(audio)) < len(audio)*1e-4:
                with open("feature_removed.txt", "a+") as text_file:
                    # print("Purchase Amount: {}".format(TotalAmount), file=text_file)
                    print(f"Silent file removed in feature extractor: {audio_fn}", 
                        file=text_file)
                    tqdm.write("Silent file removed in feature extractor: {}".format(audio_fn))
                continue

            # features
            feature = RT_preprocessing(extractor, audio)
            '''(channels, time, frequency)'''               

            meta_fn = fn + '.csv'
            df = pd.read_csv(os.path.join(meta_dir, meta_fn))

            target_event = df['sound_event_recording'].values
            target_start_time = df['start_time'].values
            target_end_time = df['end_time'].values
            target_ele = df['ele'].values
            target_azi = df['azi'].values
            target_dist = df['dist'].values

            hdf5_path = os.path.join(hdf5_dir, fn + '.h5')
            with h5py.File(hdf5_path, 'w') as hf:

                hf.create_dataset('feature', data=feature, dtype=np.float32)
                # hf.create_dataset('filename', data=[na.encode() for na in [fn]], dtype='S20')

                hf.create_group('target')
                hf['target'].create_dataset('event', data=[e.encode() for e in target_event], dtype='S20')
                hf['target'].create_dataset('start_time', data=target_start_time, dtype=np.float32)
                hf['target'].create_dataset('end_time', data=target_end_time, dtype=np.float32)
                hf['target'].create_dataset('elevation', data=target_ele, dtype=np.float32)
                hf['target'].create_dataset('azimuth', data=target_azi, dtype=np.float32)
                hf['target'].create_dataset('distance', data=target_dist, dtype=np.float32)    

            tqdm.write('{}, {}, {}'.format(audio_count, hdf5_path, feature.shape))
    
    iterator.close()
    print("Extacting feature finished! Time spent: {:.3f} s".format(timer() - begin_time))


def fit(args):
    """
    Calculate scalar.

    Args:
        feature_dir: feature path
        audio_type: 'foa' | 'mic' | 'foa&mic'
    """
    
    hdf5_dir = os.path.join(args.feature_dir, args.feature_type,
                            hdf5_folder_name, args.audio_type + '_dev')

    scalar_path = os.path.join(args.feature_dir, args.feature_type,
                            hdf5_folder_name, args.audio_type + '_scalar.h5')

    os.makedirs(os.path.dirname(scalar_path), exist_ok=True)

    print('\n============> Start Calculating Scalar.\n')

    load_time = timer()
    features = []
    for hdf5_fn in os.listdir(hdf5_dir):
        hdf5_path = os.path.join(hdf5_dir, hdf5_fn)
        with h5py.File(hdf5_path, 'r') as hf:
            features.append(hf['feature'][:])
    print('Load feature time: {:.3f} s'.format(timer() - load_time))

    features = np.concatenate(features, axis=1)
    (mean, std) = calculate_scalar(features)

    with h5py.File(scalar_path, 'w') as hf_scalar:
        hf_scalar.create_dataset('mean', data=mean, dtype=np.float32)
        hf_scalar.create_dataset('std', data=std, dtype=np.float32)

    print('Features shape: {}'.format(features.shape))
    print('mean {}:\n{}'.format(mean.shape, mean))
    print('std {}:\n{}'.format(std.shape, std))
    print('Write out scalar to {}'.format(scalar_path))


def extract_eval_features(args):
    """
    Write features and infos of audios to hdf5.

    Args:
        dataset_dir: dataset path
        feature_dir: feature path
        audio_type: 'foa' | 'mic' | 'foa&mic'
    """
    # extractor
    if args.feature_type == 'logmel':
        extractor = LogMelExtractor(fs=fs, 
                                    nfft=nfft,
                                    hopsize=hopsize,
                                    mel_bins=mel_bins,
                                    window=window,
                                    fmin=fmin)  
    elif args.feature_type == 'logmelgcc':
        extractor = LogMelGccExtractor(fs=fs, 
                                    nfft=nfft,
                                    hopsize=hopsize,
                                    mel_bins=mel_bins,
                                    window=window,
                                    fmin=fmin)
    elif args.feature_type == 'logmelintensity':
        extractor = LogMelIntensityExtractor(fs=fs, 
                                    nfft=nfft,
                                    hopsize=hopsize,
                                    mel_bins=mel_bins,
                                    window=window,
                                    fmin=fmin)    
    elif args.feature_type == 'logmelgccintensity':
        extractor = LogMelGccIntensityExtractor(fs=fs, 
                                    nfft=nfft,
                                    hopsize=hopsize,
                                    mel_bins=mel_bins,
                                    window=window,
                                    fmin=fmin)

    # Path
    if args.feature_type == 'logmelgccintensity':
        audio_dir = [os.path.join(args.dataset_dir, 'eval', 'foa_eval'), os.path.join(args.dataset_dir, 'eval', 'mic_eval')]
        hdf5_dir = os.path.join(args.feature_dir, args.feature_type,
                                hdf5_folder_name, 'foa&mic_eval')
        os.makedirs(hdf5_dir, exist_ok=True)
    else:
        audio_dir = [os.path.join(args.dataset_dir,  'eval', args.audio_type + '_eval')]
        hdf5_dir = os.path.join(args.feature_dir, args.feature_type,
                                hdf5_folder_name, args.audio_type + '_eval')
        os.makedirs(hdf5_dir, exist_ok=True)

    begin_time = timer()
    audio_count = 0
                                       
    print('\n============> Start Extracting Features\n')

    iterator = tqdm(sorted(os.listdir(audio_dir[0])), total=len(os.listdir(audio_dir[0])), unit='it')

    for audio_fn in iterator:

        if audio_fn.endswith('.wav') and not audio_fn.startswith('.'):

            fn = audio_fn.split('.')[0]
            if args.feature_type == 'logmelgccintensity':
                audio_path = [os.path.join(audio_dir[0], audio_fn), os.path.join(audio_dir[1], audio_fn)]
                audio_foa, _ = librosa.load(audio_path[0], sr=fs, mono=False, dtype=np.float32)
                audio_mic, _ = librosa.load(audio_path[1], sr=fs, mono=False, dtype=np.float32)
                audio_len = min(audio_foa.shape[1], audio_mic.shape[1])
                audio = np.concatenate([audio_foa[:, :audio_len], audio_mic[:, :audio_len]], axis=0)
                '''(channel_nums, samples)'''
            else:
                audio_path = os.path.join(audio_dir[0], audio_fn)
                audio, _ = librosa.load(audio_path, sr=fs, mono=False, dtype=np.float32)
                '''(channel_nums, samples)'''

            audio_count += 1           

            if np.sum(np.abs(audio)) < len(audio)*1e-4:
                with open("feature_removed.txt", "a+") as text_file:
                    print(f"Silent file removed in feature extractor: {audio_fn}", 
                        file=text_file)
                    tqdm.write("Silent file removed in feature extractor: {}".format(audio_fn))
                continue

            # features
            feature = RT_preprocessing(extractor, audio)
            '''(channels, time, frequency)'''       

            hdf5_path = os.path.join(hdf5_dir, fn + '.h5')
            with h5py.File(hdf5_path, 'w') as hf:

                hf.create_dataset('feature', data=feature, dtype=np.float32)

            tqdm.write('{}, {}, {}'.format(audio_count, hdf5_path, feature.shape))
    
    iterator.close()
    print("Extacting feature finished! Time spent: {:.3f} s".format(timer() - begin_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features from audio file')

    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--feature_dir', type=str, required=True)
    parser.add_argument('--feature_type', type=str, required=True,
                                choices=['logmel', 'logmelgcc', 'logmelintensity', 'logmelgccintensity'])   
    parser.add_argument('--data_type', type=str, required=True, 
                                choices=['dev', 'eval'])
    parser.add_argument('--audio_type', type=str, required=True,
                                choices=['foa', 'mic', 'foa&mic'])

    args = parser.parse_args()

    if args.feature_type == 'logmelgccintensity':
        args.audio_type = 'foa&mic'

    if args.data_type == 'dev':
        extract_dev_features(args)
        fit(args)
    elif args.data_type == 'eval':
        extract_eval_features(args)
