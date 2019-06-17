import argparse
import os
import pdb
import shutil
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from tqdm import tqdm

from evaluation import write_submission


def iters_ensemble(args):
    '''

    Ensemble on different iterations and generate ensembled files in fusioned folder
    '''
    ## directories
    if args.task_type == 'sed_only':
        # iterations ensemble directory
        fusioned_dir = os.path.join(submissions_dir, args.name + '_' + args.model_sed + '_{}'.format(args.audio_type) + '_{}'.format(args.feature_type) +
                '_aug_{}'.format(args.data_aug) + '_seed_{}'.format(args.seed), 'sed_mask_fusioned')
        os.makedirs(fusioned_dir, exist_ok=True)

        fusion_fn = '_fusion_sed_epoch_{}'
        iterator = range(38, 42, 2)

    elif args.task_type == 'two_staged_eval':
        # iterations ensemble directory
        fusioned_dir = os.path.join(submissions_dir, args.name + '_' + args.model_sed + '_{}'.format(args.audio_type) + '_{}'.format(args.feature_type) +
                '_aug_{}'.format(args.data_aug) + '_seed_{}'.format(args.seed), 'doa_fusioned')
        os.makedirs(fusioned_dir, exist_ok=True)

        fusion_fn = '_fusion_doa_epoch_{}'
        iterator = range(78, 82, 2)

    ## average ensemble
    print('\n===> Average ensemble')
    ensemble_start_time = timer()
    predicts_fusioned = []
    for epoch_num in iterator:
        fusion_dir = os.path.join(submissions_dir, args.name + '_' + args.model_sed + '_{}'.format(args.audio_type) + '_{}'.format(args.feature_type) +
            '_aug_{}'.format(args.data_aug) + '_seed_{}'.format(args.seed), fusion_fn.format(epoch_num))

        for fn in sorted(os.listdir(fusion_dir)):
            if fn.endswith('.csv') and not fn.startswith('.'):
                fn_path = os.path.join(fusion_dir, fn)
                predicts_fusioned.append(pd.read_csv(fn_path, header=0, index_col=0).values)
        
        if len(predicts_fusioned) > file_num:
            for n in range(file_num):
                min_len = min(predicts_fusioned[n].shape[0], predicts_fusioned[n+file_num].shape[0])
                predicts_fusioned[n] = (predicts_fusioned[n][:min_len,:] + predicts_fusioned[n+file_num][:min_len,:]) / 2
            predicts_fusioned = predicts_fusioned[:file_num]
    print('\nAverage ensemble time: {:.3f} s.'.format(timer()-ensemble_start_time))

    ## write the fusioned sed probabilities or doa predictions to fusioned files
    print('\n===> Write the fusioned sed probabilities or doa predictions to fusioned files')
    # this folder here is only used for supplying fn
    iterator = tqdm(sorted(os.listdir(fusion_dir)), total=len(os.listdir(fusion_dir)), unit='iters')

    n = 0
    for fn in iterator:
        if fn.endswith('.csv') and not fn.startswith('.'):
            # write to sed_mask_fusioned folder
            fn_path = os.path.join(fusioned_dir, fn)
            df_output = pd.DataFrame(predicts_fusioned[n])
            df_output.to_csv(fn_path)
            n += 1

    iterator.close()

    print('\n' + fusioned_dir)

    print('\n===> Iterations ensemble finished!')


def threshold_iters_ensemble(args):
    '''

    Threshold the ensembled iterations and write to submissions
    '''
    # directories
    sed_mask_fusioned_dir = os.path.join(submissions_dir, args.name + '_' + args.model_sed + '_{}'.format(args.audio_type) + '_{}'.format(args.feature_type) +
            '_aug_{}'.format(args.data_aug) + '_seed_{}'.format(args.seed), 'sed_mask_fusioned')
    doa_fusioned_dir = os.path.join(submissions_dir, args.name + '_' + args.model_sed + '_{}'.format(args.audio_type) + '_{}'.format(args.feature_type) +
            '_aug_{}'.format(args.data_aug) + '_seed_{}'.format(args.seed), 'doa_fusioned')

    if args.task_type == 'sed_only':
        test_fusioned_dir = os.path.join(submissions_dir, args.name + '_' + args.model_sed + '_{}'.format(args.audio_type) + '_{}'.format(args.feature_type) +
            '_aug_{}'.format(args.data_aug) + '_seed_{}'.format(args.seed), 'sed_test_fusioned')
    elif args.task_type == 'two_staged_eval':
        test_fusioned_dir = os.path.join(submissions_dir, args.name + '_' + args.model_sed + '_{}'.format(args.audio_type) + '_{}'.format(args.feature_type) +
            '_aug_{}'.format(args.data_aug) + '_seed_{}'.format(args.seed), 'all_test_fusioned')
    os.makedirs(test_fusioned_dir, exist_ok=True)

    if args.task_type == 'sed_only':
        iterator = tqdm(sorted(os.listdir(sed_mask_fusioned_dir)), total=len(os.listdir(sed_mask_fusioned_dir)), unit='iters')
        for fn in iterator:
            if fn.endswith('_prob.csv') and not fn.startswith('.'):
                fn_path = os.path.join(sed_mask_fusioned_dir, fn)
                prob_fusioned = pd.read_csv(fn_path, header=0, index_col=0).values

                # write to sed_test_fusioned
                fn_noextension = fn.split('_prob')[0]
                output_doas = np.zeros((prob_fusioned.shape[0],22))
                submit_dict = {
                    'filename': fn_noextension,
                    'events': (prob_fusioned>args.threshold).astype(np.float32),
                    'doas': output_doas
                }
                write_submission(submit_dict, test_fusioned_dir)
                
    if args.task_type == 'two_staged_eval':
        iterator = tqdm(sorted(os.listdir(doa_fusioned_dir)), total=len(os.listdir(doa_fusioned_dir)), unit='iters')
        for fn in iterator:
            if fn.endswith('_doa.csv') and not fn.startswith('.'):
                fn_noextension = fn.split('_doa')[0]

                # read sed predictions from sed_mask_fusioned directory
                fn_path = os.path.join(sed_mask_fusioned_dir, fn_noextension + '_prob.csv')
                prob_fusioned = pd.read_csv(fn_path, header=0, index_col=0).values                
                # read doa predictions from doa_fusioned directory
                fn_path = os.path.join(doa_fusioned_dir, fn)
                doa_fusioned = pd.read_csv(fn_path, header=0, index_col=0).values  
                
                # write to all_test_fusioned
                submit_dict = {
                    'filename': fn_noextension,
                    'events': (prob_fusioned>args.threshold).astype(np.float32),
                    'doas': doa_fusioned
                } 
                write_submission(submit_dict, test_fusioned_dir)

    iterator.close()

    print('\n' + test_fusioned_dir)

    print('\n===> Threshold iterations ensemble finished!')
        

def models_ensemble(args):
    '''

    Ensemble on different iterations and generate ensembled files in fusioned folder
    '''
    # directories
    if args.task_type == 'sed_only':
        fusion_folder = 'sed_mask_fusioned'
        fusioned_folder = 'sed_mask_models_fusioned'
    elif args.task_type == 'two_staged_eval':
        fusion_folder = 'doa_fusioned'
        fusioned_folder = 'doa_models_fusioned'
    
    print('\n===> Model average ensemble')
    ensemble_start_time = timer()
    predicts_fusioned = []
    for model_folder in sorted(os.listdir(submissions_dir)):
        if not model_folder.startswith('.') and model_folder != 'models_ensemble':
            print('\n' + model_folder)
            fusion_dir = os.path.join(submissions_dir, model_folder, fusion_folder)

            for fn in sorted(os.listdir(fusion_dir)):
                if fn.endswith('.csv') and not fn.startswith('.'):
                    fn_path = os.path.join(fusion_dir, fn)
                    predicts_fusioned.append(pd.read_csv(fn_path, header=0, index_col=0).values)

            if len(predicts_fusioned) > file_num:
                for n in range(file_num):
                    min_len = min(predicts_fusioned[n].shape[0], predicts_fusioned[n+file_num].shape[0])
                    predicts_fusioned[n] = (predicts_fusioned[n][:min_len,:] + predicts_fusioned[n+file_num][:min_len,:]) / 2
                predicts_fusioned = predicts_fusioned[:file_num]
    print('\nAverage ensemble time: {:.3f} s.'.format(timer()-ensemble_start_time))
    
    ## write the fusioned sed probabilities or doa predictions to fusioned files
    print('\n===> Write the fusioned sed probabilities or doa predictions to fusioned files')
    # this folder here is only used for supplying fn
    iterator = tqdm(sorted(os.listdir(fusion_dir)), total=len(os.listdir(fusion_dir)), unit='iters')

    models_ensemble_dir = os.path.join(submissions_dir, 'models_ensemble', fusioned_folder)
    os.makedirs(models_ensemble_dir, exist_ok=True)

    n = 0
    for fn in iterator:
        if fn.endswith('.csv') and not fn.startswith('.'):
            # write to sed_mask_fusioned folder
            fn_path = os.path.join(models_ensemble_dir, fn)
            df_output = pd.DataFrame(predicts_fusioned[n])
            df_output.to_csv(fn_path)
            n += 1

    iterator.close()

    print('\n' + models_ensemble_dir)

    print('\n===> Models ensemble finished!')
        

def threshold_models_ensemble(args):
    '''

    Threshold the ensembled models and write to submissions
    '''
    # directories
    sed_mask_fusioned_dir = os.path.join(submissions_dir, 'models_ensemble', 'sed_mask_models_fusioned')
    doa_fusioned_dir = os.path.join(submissions_dir, 'models_ensemble', 'doa_models_fusioned')

    if args.task_type == 'sed_only':
        test_fusioned_dir = os.path.join(submissions_dir, args.name + '_' + args.model_sed + '_{}'.format(args.audio_type) + '_{}'.format(args.feature_type) +
            '_aug_{}'.format(args.data_aug) + '_seed_{}'.format(args.seed), 'sed_test_fusioned')
    elif args.task_type == 'two_staged_eval':
        test_fusioned_dir = os.path.join(submissions_dir, args.name + '_' + args.model_sed + '_{}'.format(args.audio_type) + '_{}'.format(args.feature_type) +
            '_aug_{}'.format(args.data_aug) + '_seed_{}'.format(args.seed), 'all_test_fusioned')
    os.makedirs(test_fusioned_dir, exist_ok=True)

    if args.task_type == 'sed_only':
        iterator = tqdm(sorted(os.listdir(sed_mask_fusioned_dir)), total=len(os.listdir(sed_mask_fusioned_dir)), unit='iters')
        for fn in iterator:
            if fn.endswith('_prob.csv') and not fn.startswith('.'):
                fn_path = os.path.join(sed_mask_fusioned_dir, fn)
                prob_fusioned = pd.read_csv(fn_path, header=0, index_col=0).values

                # write to sed_test_fusioned
                fn_noextension = fn.split('_prob')[0]
                output_doas = np.zeros((prob_fusioned.shape[0],22))
                submit_dict = {
                    'filename': fn_noextension,
                    'events': (prob_fusioned>args.threshold).astype(np.float32),
                    'doas': output_doas
                }
                write_submission(submit_dict, test_fusioned_dir)
                
    if args.task_type == 'two_staged_eval':
        iterator = tqdm(sorted(os.listdir(doa_fusioned_dir)), total=len(os.listdir(doa_fusioned_dir)), unit='iters')
        for fn in iterator:
            if fn.endswith('_doa.csv') and not fn.startswith('.'):
                fn_noextension = fn.split('_doa')[0]

                # read sed predictions from sed_mask_fusioned directory
                fn_path = os.path.join(sed_mask_fusioned_dir, fn_noextension + '_prob.csv')
                prob_fusioned = pd.read_csv(fn_path, header=0, index_col=0).values                
                # read doa predictions from doa_fusioned directory
                fn_path = os.path.join(doa_fusioned_dir, fn)
                doa_fusioned = pd.read_csv(fn_path, header=0, index_col=0).values  
                
                # write to all_test_fusioned
                submit_dict = {
                    'filename': fn_noextension,
                    'events': (prob_fusioned>args.threshold).astype(np.float32),
                    'doas': doa_fusioned
                } 
                write_submission(submit_dict, test_fusioned_dir)

    iterator.close()

    print('\n' + test_fusioned_dir)

    print('\n===> Threshold models ensemble finished!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ensemble on different iterations or different models')

    subparsers = parser.add_subparsers(dest='mode')

    parser_iters_ensemble = subparsers.add_parser('iters_ensemble')
    parser_iters_ensemble.add_argument('--workspace', type=str, required=True,
                                help='workspace directory')
    parser_iters_ensemble.add_argument('--feature_type', type=str, required=True,
                                choices=['logmel', 'logmelgcc', 'logmelintensity', 'logmelgccintensity'])
    parser_iters_ensemble.add_argument('--audio_type', type=str, required=True, 
                                choices=['foa', 'mic', 'foa&mic'], help='audio type')
    parser_iters_ensemble.add_argument('--task_type', type=str, required=True,
                                choices=['sed_only', 'doa_only', 'two_staged_eval', 'seld'])                          
    parser_iters_ensemble.add_argument('--model_sed', type=str, default='CRNN10')
    parser_iters_ensemble.add_argument('--model_doa', type=str, default='pretrained_CRNN10')    
    parser_iters_ensemble.add_argument('--data_aug', default='None', type=str,
                                help='data augmentation methods')                         
    parser_iters_ensemble.add_argument('--seed', default=42, type=int, help='random seed')
    parser_iters_ensemble.add_argument('--name', default='n0', type=str)

    parser_threshold_iters_ensemble = subparsers.add_parser('threshold_iters_ensemble')
    parser_threshold_iters_ensemble.add_argument('--workspace', type=str, required=True,
                                help='workspace directory')
    parser_threshold_iters_ensemble.add_argument('--feature_type', type=str, required=True,
                                choices=['logmel', 'logmelgcc', 'logmelintensity', 'logmelgccintensity'])
    parser_threshold_iters_ensemble.add_argument('--audio_type', type=str, required=True, 
                                choices=['foa', 'mic', 'foa&mic'], help='audio type')
    parser_threshold_iters_ensemble.add_argument('--task_type', type=str, required=True,
                                choices=['sed_only', 'doa_only', 'two_staged_eval', 'seld'])                          
    parser_threshold_iters_ensemble.add_argument('--model_sed', type=str, default='CRNN10')
    parser_threshold_iters_ensemble.add_argument('--model_doa', type=str, default='pretrained_CRNN10')    
    parser_threshold_iters_ensemble.add_argument('--data_aug', default='None', type=str,
                                help='data augmentation methods')                         
    parser_threshold_iters_ensemble.add_argument('--seed', default=42, type=int, help='random seed')
    parser_threshold_iters_ensemble.add_argument('--name', default='n0', type=str)
    parser_threshold_iters_ensemble.add_argument('--threshold', default=0.3, type=float)

    parser_models_ensemble = subparsers.add_parser('models_ensemble')
    parser_models_ensemble.add_argument('--workspace', type=str, required=True,
                                help='workspace directory')
    parser_models_ensemble.add_argument('--feature_type', type=str, required=True,
                                choices=['logmel', 'logmelgcc', 'logmelintensity', 'logmelgccintensity'])
    parser_models_ensemble.add_argument('--audio_type', type=str, required=True, 
                                choices=['foa', 'mic', 'foa&mic'], help='audio type')
    parser_models_ensemble.add_argument('--task_type', type=str, required=True,
                                choices=['sed_only', 'doa_only', 'two_staged_eval', 'seld'])                          
    parser_models_ensemble.add_argument('--model_sed', type=str, default='CRNN10')
    parser_models_ensemble.add_argument('--model_doa', type=str, default='pretrained_CRNN10')    
    parser_models_ensemble.add_argument('--data_aug', default='None', type=str,
                                help='data augmentation methods')                         
    parser_models_ensemble.add_argument('--seed', default=42, type=int, help='random seed')
    parser_models_ensemble.add_argument('--name', default='n0', type=str)

    parser_threshold_models_ensemble = subparsers.add_parser('threshold_models_ensemble')
    parser_threshold_models_ensemble.add_argument('--workspace', type=str, required=True,
                                help='workspace directory')
    parser_threshold_models_ensemble.add_argument('--feature_type', type=str, required=True,
                                choices=['logmel', 'logmelgcc', 'logmelintensity', 'logmelgccintensity'])
    parser_threshold_models_ensemble.add_argument('--audio_type', type=str, required=True, 
                                choices=['foa', 'mic', 'foa&mic'], help='audio type')
    parser_threshold_models_ensemble.add_argument('--task_type', type=str, required=True,
                                choices=['sed_only', 'doa_only', 'two_staged_eval', 'seld'])                          
    parser_threshold_models_ensemble.add_argument('--model_sed', type=str, default='CRNN10')
    parser_threshold_models_ensemble.add_argument('--model_doa', type=str, default='pretrained_CRNN10')    
    parser_threshold_models_ensemble.add_argument('--data_aug', default='None', type=str,
                                help='data augmentation methods')                         
    parser_threshold_models_ensemble.add_argument('--seed', default=42, type=int, help='random seed')
    parser_threshold_models_ensemble.add_argument('--name', default='n0', type=str)
    parser_threshold_models_ensemble.add_argument('--threshold', default=0.3, type=float)

    args = parser.parse_args()

    # submissions directory
    global submissions_dir
    submissions_dir = os.path.join(args.workspace, 'appendixes', 'submissions_eval')

    global file_num
    file_num = 100

    # ensemble different iterations or models
    if args.mode == 'iters_ensemble':
        iters_ensemble(args)
    elif args.mode == 'threshold_iters_ensemble':
        threshold_iters_ensemble(args)
    elif args.mode == 'models_ensemble':
        models_ensemble(args)
    elif args.mode == 'threshold_models_ensemble':
        threshold_models_ensemble(args)