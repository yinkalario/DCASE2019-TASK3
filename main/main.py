import argparse
import logging
import os
import pdb
import random
import shutil
import socket
import sys
from datetime import datetime
from timeit import default_timer as timer
sys.path.insert(1, os.path.join(sys.path[0], '../'))

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import average_precision_score, confusion_matrix
from tensorboardX import SummaryWriter
# from torch.utils.tensorboard.writer.SummaryWriter import SummaryWriter
from torch.backends import cudnn
from tqdm import tqdm

import evaluation
import models
from loss import hybrid_regr_loss
from torchsummary import summary
from utils.data_augmentation import freq_mask, mixup, time_mask
from utils.data_generator import DevDataGenerator, EvalDataGenerator
from utils.utilities import (create_logging, doa_labels, event_labels,
                             get_filename, logging_and_writer,
                             move_model_to_gpu, print_evaluation, str2bool,
                             to_np, to_torch)


## Hyper-parameters
################# Model #################
model_pool_type = 'avg'         # 'max' | 'avg'
model_pool_size = (2,2)

loss_type = 'MAE'
################# param #################
weight_decay = 0
threshold = {'sed': 0.5}

fs = 32000
nfft = 1024
hopsize = 320 # 640 for 20 ms
mel_bins = 128
frames_per_1s = fs // hopsize
sub_frames_per_1s = 50
hopframes = int(0.5 * frames_per_1s)
hdf5_folder_name = '{}fs_{}nfft_{}hs_{}melb'.format(fs, nfft, hopsize, mel_bins)


def train(args, data_generator, model, optimizer, initial_epoch, logging):
    '''

    Train goes here
    '''
    # set up tensorboard writer
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    tb_dir = os.path.join(
        'runs', datetime.now().strftime('%b%d'), args.task_type, args.name + '_' + 
            args.model + '_{}'.format(args.audio_type) + '_{}'.format(args.feature_type) + '_aug_{}'.format(args.data_aug) + 
            '_fold_{}'.format(args.fold) + '_seed_{}'.format(args.seed), current_time + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=tb_dir)
    writer.add_text('Parameters', str(args))

    temp_submissions_dir_train = os.path.join(temp_submissions_trial_dir, 'train')
    temp_submissions_dir_valid = os.path.join(temp_submissions_trial_dir, 'valid')

    logging.info('\n===> Training mode')

    train_begin_time = timer()

    batchNum_per_epoch = data_generator.batchNum_per_epoch

    iterator = tqdm(enumerate(data_generator.generate_train()),
        total=args.max_epochs*batchNum_per_epoch, unit='batch')

    for batch_idx, (batch_x, batch_y_dict) in iterator:

        epoch = int(batch_idx//batchNum_per_epoch) + initial_epoch
        batch_epoch = int(batch_idx%batchNum_per_epoch)

        ################
        ## Validation
        ################
        if batch_idx % 200 == 0:

            valid_begin_time = timer()
            train_time = valid_begin_time - train_begin_time

            # Train evaluation
            shutil.rmtree(temp_submissions_dir_train, ignore_errors=True)
            os.makedirs(temp_submissions_dir_train, exist_ok=False)
            train_metrics = evaluation.evaluate(
                        data_generator=data_generator, 
                        data_type='train', 
                        max_audio_num=30,
                        task_type=args.task_type, 
                        model=model, 
                        cuda=args.cuda,
                        loss_type=loss_type,
                        threshold=threshold,
                        submissions_dir=temp_submissions_dir_train, 
                        frames_per_1s=frames_per_1s,
                        sub_frames_per_1s=sub_frames_per_1s)

            logging.info('------------------------------------------------------------------------------------------------------------------------------------')
            
            # Validation evaluation
            if args.fold != -1:
                shutil.rmtree(temp_submissions_dir_valid, ignore_errors=True)
                os.makedirs(temp_submissions_dir_valid, exist_ok=False)
                valid_metrics = evaluation.evaluate(
                        data_generator=data_generator, 
                        data_type='valid', 
                        max_audio_num=30,
                        task_type=args.task_type, 
                        model=model, 
                        cuda=args.cuda,
                        loss_type=loss_type,
                        threshold=threshold, 
                        submissions_dir=temp_submissions_dir_valid, 
                        frames_per_1s=frames_per_1s, 
                        sub_frames_per_1s=sub_frames_per_1s)
                metrics = [train_metrics, valid_metrics]
                logging_and_writer('valid', metrics, logging, writer, batch_idx)
            else:
                logging_and_writer('train', train_metrics, logging, writer, batch_idx)

            valid_time = timer() - valid_begin_time
            logging.info('Iters: {},  Epoch/Total epoch: {}/{},  Batch/Total batch per epoch: {}/{},  Train time: {:.3f}s,  Eval time: {:.3f}s'.format(
                        batch_idx, epoch, args.max_epochs+initial_epoch, batch_epoch, batchNum_per_epoch, train_time, valid_time))             
            logging.info('------------------------------------------------------------------------------------------------------------------------------------')
            train_begin_time = timer()

        ###############
        ## Save model
        ###############
        if args.task_type == 'sed_only':
            save_factor = 38 # 12
        elif args.task_type == 'doa_only':
            save_factor = 78 # 48
        if ((batch_idx % (2*batchNum_per_epoch)) == 0) and ((batch_idx + initial_epoch*batchNum_per_epoch) >= save_factor*batchNum_per_epoch):
            save_path = os.path.join(models_dir, 'epoch_{}.pth'.format(epoch))
            checkpoint = {'model_state_dict': model.module.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(),
                          'rng_state': torch.get_rng_state(),
                          'cuda_rng_state': torch.cuda.get_rng_state(),
                          'epoch': epoch}
            torch.save(checkpoint, save_path)
            logging.info('Checkpoint saved to {}'.format(save_path))

        ###############
        ## Train
        ###############
        # Reduce learning rate
        if args.task_type == 'sed_only':
            reduce_factor = 26 # 12
        elif args.task_type == 'doa_only':
            reduce_factor = 48 # 48
        if args.reduce_lr:
            # if ((batch_idx % (batchNum_per_epoch)) == 0) and (batch_idx >= reduce_factor*batchNum_per_epoch):
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] *= 0.9
            if (batch_idx + initial_epoch*batchNum_per_epoch) >= reduce_factor*batchNum_per_epoch:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.0001

        batch_x = to_torch(batch_x, args.cuda)
        batch_y_dict = {
            'events':   to_torch(batch_y_dict['events'], args.cuda),
            'doas':  to_torch(batch_y_dict['doas'], args.cuda)
        }

        if args.data_aug == 'mixup':
            batch_x, batch_y_dict['events'] = mixup(batch_x, batch_y_dict['events'], alpha=0.1)
        elif args.data_aug == 'specaug':
            batch_x = freq_mask(batch_x, ratio_F=0.1, num_masks=2, replace_with_zero=False)
            batch_x = time_mask(batch_x, ratio_T=0.1, num_masks=2, replace_with_zero=False)
        elif args.data_aug == 'mixup&specaug':
            batch_x, batch_y_dict['events'] = mixup(batch_x, batch_y_dict['events'], alpha=0.1)
            batch_x = freq_mask(batch_x, ratio_F=0.1, num_masks=2, replace_with_zero=False)
            batch_x = time_mask(batch_x, ratio_T=0.1, num_masks=2, replace_with_zero=False)

        # Forward
        model.train()
        output = model(batch_x)
        
        # Loss
        seld_loss, _, _ = hybrid_regr_loss(output, batch_y_dict, args.task_type, loss_type=loss_type)

        # Backward
        optimizer.zero_grad()
        seld_loss.backward()
        optimizer.step()

        if batch_idx == args.max_epochs*batchNum_per_epoch:
            iterator.close()
            writer.close()
            break   


def test(args, data_generator, logging):
    '''

    Test split evaluation.
    '''
    logging.info('\n===> Evaluate test split fold {}'.format(args.fold))

    if args.task_type == 'sed_only':
        task_type = 'sed_only'
        submissions_model = args.model_sed
        logging.info('\n===> Test SED')
    elif args.task_type == 'seld':
        task_type = 'seld'
        submissions_model = args.model_sed
        logging.info('\n===> Test SELD')
    elif args.task_type == 'doa_only':
        task_type = 'doa_only'
        submissions_model = args.model_doa
        logging.info('\n===> Test DOA')
    elif args.task_type == 'two_staged_eval':
        task_type = 'doa_only'
        submissions_model = args.model_sed
        logging.info('\n===> Two Staged Eval')

    # Inference
    model_path = os.path.join(models_dir, task_type, args.name + '_' + 
                            args.model + '_{}'.format(args.audio_type) + '_{}'.format(args.feature_type) + '_aug_{}'.format(args.data_aug) + 
                            '_fold_{}'.format(args.fold) + '_seed_{}'.format(args.seed), 'epoch_{}.pth'.format(args.epoch_num))
    assert os.path.exists(model_path), 'Error: no checkpoint file found!'
    model = models.__dict__[args.model](class_num, args.model_pool_type, args.model_pool_size, None)
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model_state_dict'])
    if args.cuda:
        model.cuda()

    fold_submissions_dir= os.path.join(submissions_dir, args.name + '_' + submissions_model + '_{}'.format(args.audio_type) + '_{}'.format(args.feature_type) +
        '_aug_{}'.format(args.data_aug) + '_seed_{}'.format(args.seed), '_test')
    shutil.rmtree(fold_submissions_dir, ignore_errors=True)
    os.makedirs(fold_submissions_dir, exist_ok=False)

    test_metrics = evaluation.evaluate(
            data_generator=data_generator, 
            data_type='test', 
            max_audio_num=None,
            task_type=args.task_type, 
            model=model, 
            cuda=args.cuda,
            loss_type=loss_type,
            threshold=threshold,
            submissions_dir=fold_submissions_dir, 
            frames_per_1s=frames_per_1s,
            sub_frames_per_1s=sub_frames_per_1s,
            FUSION=args.fusion,
            epoch_num=args.epoch_num)

    logging.info('------------------------------------------------------------------------------------------------------------------------------------')
    logging_and_writer('test', test_metrics, logging)
    logging.info('------------------------------------------------------------------------------------------------------------------------------------')

    if args.task_type == 'sed_only':
        test_submissions_dir= os.path.join(submissions_dir, args.name + '_' + args.model_sed + '_{}'.format(args.audio_type) + '_{}'.format(args.feature_type) + 
            '_aug_{}'.format(args.data_aug) + '_seed_{}'.format(args.seed), 'sed_test')
    elif args.task_type == 'two_staged_eval':
        test_submissions_dir= os.path.join(submissions_dir, args.name + '_' + args.model_sed + '_{}'.format(args.audio_type) + '_{}'.format(args.feature_type) + 
            '_aug_{}'.format(args.data_aug) + '_seed_{}'.format(args.seed), 'all_test')        
    os.makedirs(test_submissions_dir, exist_ok=True)
    for fn in sorted(os.listdir(fold_submissions_dir)):
        if fn.endswith('.csv') and not fn.startswith('.'):
            src = os.path.join(fold_submissions_dir, fn)
            dst = os.path.join(test_submissions_dir, fn)
            shutil.copyfile(src, dst)


def test_all_folds(args):
    '''

    Evaluate all test split folds.
    '''
    if args.fusion:
        if args.task_type == 'sed_only':
            test_submissions_dir = os.path.join(args.workspace, 'appendixes', 'submissions', args.name + '_' + args.model_sed + '_{}'.format(args.audio_type) +
                '_{}'.format(args.feature_type) + '_aug_{}'.format(args.data_aug) + '_seed_{}'.format(args.seed), 'sed_test_fusioned') 
        elif args.task_type == 'two_staged_eval':
            test_submissions_dir = os.path.join(args.workspace, 'appendixes', 'submissions', args.name + '_' + args.model_sed + '_{}'.format(args.audio_type) +
                '_{}'.format(args.feature_type) + '_aug_{}'.format(args.data_aug) + '_seed_{}'.format(args.seed), 'all_test_fusioned')
    else:
        if args.task_type == 'sed_only':
            test_submissions_dir = os.path.join(args.workspace, 'appendixes', 'submissions', args.name + '_' + args.model_sed + '_{}'.format(args.audio_type) +
                '_{}'.format(args.feature_type) + '_aug_{}'.format(args.data_aug) + '_seed_{}'.format(args.seed), 'sed_test') 
        elif args.task_type == 'two_staged_eval':
            test_submissions_dir = os.path.join(args.workspace, 'appendixes', 'submissions', args.name + '_' + args.model_sed + '_{}'.format(args.audio_type) +
                '_{}'.format(args.feature_type) + '_aug_{}'.format(args.data_aug) + '_seed_{}'.format(args.seed), 'all_test')        

    gt_meta_dir = '/vol/vssp/AP_datasets/audio/dcase2019/task3/dataset_root/dev/metadata_dev/'

    print('\n===> Test All folds')
    start_time = timer()
    sed_scores, doa_er_metric, seld_metric = evaluation.calculate_SELD_metrics(gt_meta_dir, test_submissions_dir, score_type='all')

    loss = [0.0, 0.0, 0.0]
    sed_mAP = [0.0, 0.0]

    metrics = [loss, sed_mAP, sed_scores, doa_er_metric, seld_metric]

    print('------------------------------------------------------------------------------------------------------------------------------------')
    print_evaluation(metrics)
    print('------------------------------------------------------------------------------------------------------------------------------------')
    print('\nTime spent: {:.3f} s.'.format(timer()-start_time))


def infer_eval(args, data_generator, logging):
    '''

    Inference evaluation data.
    '''
    logging.info('\n===> Infer the evaluation data')

    # Evaluation submission directory
    submissions_eval_dir = os.path.join(appendixes_dir, 'submissions_eval')
    os.makedirs(submissions_eval_dir, exist_ok=True)

    if args.task_type == 'sed_only':
        task_type = 'sed_only'
        submissions_model = args.model_sed
        logging.info('\n===> Test SED')
    elif args.task_type == 'two_staged_eval':
        task_type = 'doa_only'
        submissions_model = args.model_sed
        logging.info('\n===> Two Staged Eval')

    # Load model
    model_path = os.path.join(models_dir, task_type, args.name + '_' + 
                            args.model + '_{}'.format(args.audio_type) + '_{}'.format(args.feature_type) + '_aug_{}'.format(args.data_aug) + 
                            '_fold_-1' + '_seed_{}'.format(args.seed), 'epoch_{}.pth'.format(args.epoch_num))
    assert os.path.exists(model_path), 'Error: no checkpoint file found!'
    model = models.__dict__[args.model](class_num, args.model_pool_type, args.model_pool_size, None)
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model_state_dict'])
    if args.cuda:
        model.cuda()
    
    fold_submissions_eval_dir= os.path.join(submissions_eval_dir, args.name + '_' + submissions_model + '_{}'.format(args.audio_type) + '_{}'.format(args.feature_type) +
        '_aug_{}'.format(args.data_aug) + '_seed_{}'.format(args.seed), '_test')
    shutil.rmtree(fold_submissions_eval_dir, ignore_errors=True)
    os.makedirs(fold_submissions_eval_dir, exist_ok=False)

    generate_func = tqdm(enumerate(data_generator.generate_eval()), total=100, unit='iter')
    start_time = timer()

    for batch_idx, (batch_x, batch_fn) in generate_func:

        batch_x = to_torch(batch_x, args.cuda)
        with torch.no_grad():
            model.eval()
            output = model(batch_x)
        output['events'] = to_np(output['events'])
        output['doas'] = to_np(output['doas'])        
        '''
        output = {
            'events',   (batch_size=1, time_len, class_num) 
            'doas'      (batch_size=1, time_len, 2*class_num) for 'regr' | 
                        (batch_size=1, time_len, ele_num*azi_num=324) for 'clas'
        }
        '''
        #############################################################################################################
        # save predicted sed results in 'sed_only' task
        # set output['events'] to ground truth sed in 'doa_only' task
        # load predicted sed results in 'two_staged_eval' task
        sed_mask_dir = os.path.join(os.path.abspath(os.path.join(fold_submissions_eval_dir, os.pardir)), '_sed_mask')
        os.makedirs(sed_mask_dir, exist_ok=True)
        hdf5_path = os.path.join(sed_mask_dir, batch_fn + '.h5')
        if args.task_type == 'sed_only':
            with h5py.File(hdf5_path, 'w') as hf:
                hf.create_dataset('sed_pred', data=output['events'], dtype=np.float32)
        elif args.task_type == 'two_staged_eval':
            with h5py.File(hdf5_path, 'r') as hf:
                output['events'] = hf['sed_pred'][:]
        #############################################################################################################

        ########################################## Interpolation ########################################
        output_events= evaluation.interp_tensor(output['events'].squeeze(), frames_per_1s, sub_frames_per_1s)
        output_doas = evaluation.interp_tensor(output['doas'].squeeze(), frames_per_1s, sub_frames_per_1s)
        #################################################################################################

        ################## Write probability and ground_truth to csv file for fusion ####################
        if args.fusion and args.task_type == 'sed_only':
            fn_prob = '{}_prob.csv'.format(batch_fn)
            fusion_sed_dir = os.path.join(os.path.abspath(os.path.join(fold_submissions_eval_dir, os.pardir)), '_fusion_sed_epoch_{}'.format(args.epoch_num))
            os.makedirs(fusion_sed_dir, exist_ok=True)
            file_path_prob = os.path.join(fusion_sed_dir, fn_prob)

            df_output = pd.DataFrame(output_events)
            df_output.to_csv(file_path_prob)

        elif args.fusion and args.task_type == 'two_staged_eval':
            fn_doa = '{}_doa.csv'.format(batch_fn)
            fusion_doa_dir = os.path.join(os.path.abspath(os.path.join(fold_submissions_eval_dir, os.pardir)), '_fusion_doa_epoch_{}'.format(args.epoch_num))
            os.makedirs(fusion_doa_dir, exist_ok=True)
            file_path_doa = os.path.join(fusion_doa_dir, fn_doa)

            df_output = pd.DataFrame(output_doas)
            df_output.to_csv(file_path_doa)
        #################################################################################################

        ############################### for submission method evaluation ################################
        if args.fusion and args.task_type == 'two_staged_eval':
            submission_models_dir = os.path.abspath(os.path.join(fold_submissions_eval_dir, os.pardir))
            fusion_dir = os.path.join(os.path.abspath(os.path.join(submission_models_dir, os.pardir)), 'models_ensemble', \
                'sed_mask_models_fusioned')

            # fusion_dir = os.path.join(os.path.abspath(os.path.join(fold_submissions_eval_dir, os.pardir)), 'sed_mask_fusioned')
            fn_path = os.path.join(fusion_dir, batch_fn+'_prob.csv')
            prob_fusioned = pd.read_csv(fn_path, header=0, index_col=0).values
            submit_dict = {
                'filename': batch_fn,
                'events': (prob_fusioned>threshold['sed']).astype(np.float32),
                'doas': output_doas
            }
        else:
            submit_dict = {
                'filename': batch_fn,
                'events': (output_events>threshold['sed']).astype(np.float32),
                'doas': output_doas
            }
        evaluation.write_submission(submit_dict, fold_submissions_eval_dir)
        #################################################################################################

    generate_func.close()

    logging.info('Evaluation time: {:.3f}s'.format(timer() - start_time))

    if args.task_type == 'sed_only':
        test_submissions_dir= os.path.join(submissions_eval_dir, args.name + '_' + args.model_sed + '_{}'.format(args.audio_type) + '_{}'.format(args.feature_type) + 
            '_aug_{}'.format(args.data_aug) + '_seed_{}'.format(args.seed), 'sed_test')
    elif args.task_type == 'two_staged_eval':
        test_submissions_dir= os.path.join(submissions_eval_dir, args.name + '_' + args.model_sed + '_{}'.format(args.audio_type) + '_{}'.format(args.feature_type) + 
            '_aug_{}'.format(args.data_aug) + '_seed_{}'.format(args.seed), 'all_test')        
    os.makedirs(test_submissions_dir, exist_ok=True)
    for fn in sorted(os.listdir(fold_submissions_eval_dir)):
        if fn.endswith('.csv') and not fn.startswith('.'):
            src = os.path.join(fold_submissions_eval_dir, fn)
            dst = os.path.join(test_submissions_dir, fn)
            shutil.copyfile(src, dst)    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DCASE2019 task3')

    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str, required=True,
                                help='workspace directory')
    parser_train.add_argument('--feature_dir', type=str, required=True,
                                help='feature directory')
    parser_train.add_argument('--feature_type', type=str, required=True,
                                choices=['logmel', 'logmelgcc', 'logmelintensity', 'logmelgccintensity'])
    parser_train.add_argument('--audio_type', type=str, required=True, 
                              choices=['foa', 'mic', 'foa&mic'], help='audio type')
    parser_train.add_argument('--task_type', type=str, required=True,
                                choices=['sed_only', 'doa_only', 'two_staged_eval', 'seld'])
    parser_train.add_argument('--batch_size', type=int, required=True,
                                help='Batch size')
    parser_train.add_argument('--max_epochs', type=int, required=True,
                                help='maximum epochs for training')
    parser_train.add_argument('--lr', default=1e-3, type=float)
    parser_train.add_argument('--reduce_lr', default=False, type=str2bool)
    parser_train.add_argument('--model_sed', type=str, default='CRNN10')
    parser_train.add_argument('--model_doa', type=str, default='pretrained_CRNN10')
    parser_train.add_argument('--fold', default=1, type=int,
                                help='fold for cross validation, if -1, use full data')
    parser_train.add_argument('--pretrained_epoch', default=28, type=int,
                                help='pretrained epoch for DOA')
    parser_train.add_argument('--resume_epoch', default=-1, type=int,
                                help='resume from epoch')
    parser_train.add_argument('--data_aug', default='none', type=str,
                                help='data augmentation methods')              
    parser_train.add_argument('--seed', default=42, type=int,
                                help='random seed')
    parser_train.add_argument('--name', default='n0', type=str)
    parser_train.add_argument('--chunklen', default=3., type=float)

    parser_test = subparsers.add_parser('test')
    parser_test.add_argument('--workspace', type=str, required=True,
                                help='workspace directory')
    parser_test.add_argument('--feature_dir', type=str, required=True,
                                help='feature directory')
    parser_test.add_argument('--feature_type', type=str, required=True,
                                choices=['logmel', 'logmelgcc', 'logmelintensity', 'logmelgccintensity'])
    parser_test.add_argument('--audio_type', type=str, required=True, 
                                choices=['foa', 'mic', 'foa&mic'], help='audio type')
    parser_test.add_argument('--task_type', type=str, required=True,
                                choices=['sed_only', 'doa_only', 'two_staged_eval', 'seld'])     
    parser_test.add_argument('--batch_size', type=int, required=True,
                                help='Batch size')
    parser_test.add_argument('--model_sed', type=str, default='CRNN10')
    parser_test.add_argument('--model_doa', type=str, default='pretrained_CRNN10')
    parser_test.add_argument('--fold', default=1, type=int,
                                help='fold for test')
    parser_test.add_argument('--epoch_num', default=24, type=int,
                                help='which epoch model to read')        
    parser_test.add_argument('--data_aug', default='None', type=str,
                                help='data augmentation methods')                         
    parser_test.add_argument('--seed', default=42, type=int, help='random seed')
    parser_test.add_argument('--name', default='n0', type=str)
    parser_test.add_argument('--fusion', type=str2bool, default=False,
                                help='Ensemble or not')
    parser_test.add_argument('--chunklen', default=3., type=float)

    parser_test_all = subparsers.add_parser('test_all')
    parser_test_all.add_argument('--workspace', type=str, required=True,
                                help='workspace directory')
    parser_test_all.add_argument('--feature_type', type=str, required=True,
                                choices=['logmel', 'logmelgcc', 'logmelintensity', 'logmelgccintensity'])
    parser_test_all.add_argument('--audio_type', type=str, required=True, 
                                choices=['foa', 'mic', 'foa&mic'], help='audio type')
    parser_test_all.add_argument('--task_type', type=str, required=True,
                                choices=['sed_only', 'doa_only', 'two_staged_eval', 'seld'])
    parser_test_all.add_argument('--model_sed', type=str, default='CRNN10')
    parser_test_all.add_argument('--model_doa', type=str, default='pretrained_CRNN10')
    parser_test_all.add_argument('--data_aug', default='None', type=str,
                                help='data augmentation methods')
    parser_test_all.add_argument('--seed', default=42, type=int,
                                help='random seed')
    parser_test_all.add_argument('--name', default='n0', type=str)
    parser_test_all.add_argument('--fusion', type=str2bool, default=False,
                                help='Ensemble or not')   

    parser_infer_eval = subparsers.add_parser('infer_eval')
    parser_infer_eval.add_argument('--workspace', type=str, required=True,
                                help='workspace directory')
    parser_infer_eval.add_argument('--feature_dir', type=str, required=True,
                                help='feature directory')
    parser_infer_eval.add_argument('--feature_type', type=str, required=True,
                                choices=['logmel', 'logmelgcc', 'logmelintensity', 'logmelgccintensity'])
    parser_infer_eval.add_argument('--audio_type', type=str, required=True, 
                                choices=['foa', 'mic', 'foa&mic'], help='audio type')
    parser_infer_eval.add_argument('--task_type', type=str, required=True,
                                choices=['sed_only', 'doa_only', 'two_staged_eval', 'seld'])
    parser_infer_eval.add_argument('--model_sed', type=str, default='CRNN10')
    parser_infer_eval.add_argument('--model_doa', type=str, default='pretrained_CRNN10')
    parser_infer_eval.add_argument('--fold', default=1, type=int,
                                help='fold for test')
    parser_infer_eval.add_argument('--epoch_num', default=24, type=int,
                                help='which epoch model to read')
    parser_infer_eval.add_argument('--data_aug', default='None', type=str,
                                help='data augmentation methods')
    parser_infer_eval.add_argument('--seed', default=42, type=int,
                                help='random seed')
    parser_infer_eval.add_argument('--name', default='n0', type=str)
    parser_infer_eval.add_argument('--fusion', type=str2bool, default=False,
                                help='Ensemble or not')   


    args = parser.parse_args()

    '''
    1. Miscellaneous
    '''
    ########## Parameters ##########
    if args.feature_type == 'logmelgccintensity':
        args.audio_type = 'foa&mic'

    args.fs = fs
    args.nfft = nfft
    args.hopsize = hopsize
    args.mel_bins = mel_bins
    if args.mode == 'train' or args.mode == 'test':
        args.chunklen = int(args.chunklen * frames_per_1s)
        args.hopframes = hopframes

    args.cuda = torch.cuda.is_available()
    args.weight_decay = weight_decay
    args.hdf5 = hdf5_folder_name

    if args.task_type == 'sed_only' or args.task_type == 'seld':
        args.model = args.model_sed
    elif args.task_type == 'doa_only' or args.task_type == 'two_staged_eval':
        args.model = args.model_doa
    args.model_pool_type = model_pool_type
    args.model_pool_size = model_pool_size
    args.loss_type = loss_type

    class_num = len(event_labels)
    doa_num = len(doa_labels)

    ########## Test all folds ##########
    if args.mode == 'test_all':
        # Run test all fold before anything
        test_all_folds(args)
        sys.exit()

    ########## Get reproducible results by manually seed the random number generator ##########
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.deterministic=True

    ########## Create directories ##########
    # logs directory
    logs_dir = os.path.join(args.workspace, 'logs', datetime.now().strftime('%b%d'), args.task_type, args.mode, args.name + '_' +
            args.model + '_{}'.format(args.audio_type) + '_{}'.format(args.feature_type) + '_aug_{}'.format(args.data_aug) + 
            '_fold_{}'.format(args.fold) + '_seed_{}'.format(args.seed))
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    # appendixes directory
    global appendixes_dir
    appendixes_dir = os.path.join(args.workspace, 'appendixes')
    os.makedirs(appendixes_dir, exist_ok=True)

    # submissions directory
    global submissions_dir
    submissions_dir = os.path.join(appendixes_dir, 'submissions')
    os.makedirs(submissions_dir, exist_ok=True)

    # Temperory submissions folder
    global temp_submissions_trial_dir
    temp_submissions_dir = os.path.join(appendixes_dir, '__submissions__')
    trial = 0
    while os.path.isdir(os.path.join(temp_submissions_dir, args.name + '_' + args.model + '_{}'.format(args.audio_type) + '_{}'.format(args.feature_type) \
        + '_aug_{}'.format(args.data_aug) + '_fold_{}'.format(args.fold) + '_seed_{}'.format(args.seed), 'trial_{}'.format(trial))):
        trial += 1
    temp_submissions_trial_dir = os.path.join(temp_submissions_dir, args.name + '_' + args.model + '_{}'.format(args.audio_type) + '_{}'.format(args.feature_type) \
        + '_aug_{}'.format(args.data_aug) + '_fold_{}'.format(args.fold) + '_seed_{}'.format(args.seed), 'trial_{}'.format(trial))
    os.makedirs(temp_submissions_trial_dir, exist_ok=True)

    # models directory
    global models_dir
    if args.mode == 'train':
        # models directory
        models_dir = os.path.join(appendixes_dir, 'models_saved', '{}'.format(args.task_type), args.name + '_' + 
                                args.model + '_{}'.format(args.audio_type) + '_{}'.format(args.feature_type) + '_aug_{}'.format(args.data_aug) + 
                                '_fold_{}'.format(args.fold) + '_seed_{}'.format(args.seed))
        os.makedirs(models_dir, exist_ok=True)
    elif args.mode == 'test' or args.mode == 'infer_eval':
        # models directory
        models_dir = os.path.join(appendixes_dir, 'models_saved')

    '''
    2. Model
    '''
    if args.mode == 'train':
        if args.resume_epoch != -1:
            # Load checkpoint
            logging.info('===> Resuming from checkpoint..')
            resume_model_path = os.path.join(models_dir, 'epoch_{}.pth'.format(args.resume_epoch))
            assert os.path.exists(resume_model_path), 'Error: no checkpoint file found!'
            checkpoint = torch.load(resume_model_path, map_location=lambda storage, loc: storage)

            model = models.__dict__[args.model](class_num, args.model_pool_type, 
                args.model_pool_size, None)
            model.load_state_dict(checkpoint['model_state_dict'])
            if args.cuda:
                model, Multi_GPU = move_model_to_gpu(model) # It is important to move the model to gpu first and then construct optimizer

            optimizer = optim.Adam(model.parameters(), lr=args.lr,
                                    betas=(0.9, 0.999), eps=1e-08, 
                                    weight_decay=weight_decay, amsgrad=True)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            torch.set_rng_state(checkpoint['rng_state'])
            torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
            initial_epoch = checkpoint['epoch']
            logging.info('Resuming complete from ' + resume_model_path)
        else:
            logging.info('\n===> Building model')

            # pretrained path
            global pretrained_path
            pretrained_path = os.path.join(appendixes_dir, 'models_saved', 'sed_only', args.name + '_' + 
                                        args.model_sed + '_{}'.format(args.audio_type) + '_{}'.format(args.feature_type) + '_aug_{}'.format(args.data_aug) + 
                                        '_fold_{}'.format(args.fold) + '_seed_{}'.format(args.seed), 'epoch_{}.pth'.format(args.pretrained_epoch))

            model = models.__dict__[args.model](class_num, args.model_pool_type, 
                args.model_pool_size, pretrained_path)
            if args.cuda:
                model, Multi_GPU = move_model_to_gpu(model) # It is important to move the model to gpu first and then construct optimizer

            optimizer = optim.Adam(model.parameters(), lr=args.lr,
                                    betas=(0.9, 0.999), eps=1e-08, 
                                    weight_decay=weight_decay, amsgrad=True)
            initial_epoch = 0
        
        # Print the model architecture and parameters
        logging.info('\nModel architectures:\n{}\n'.format(model))
        # summary(model, (256, 128))
        logging.info('\nParameters and size:')
        for n, (name, param) in enumerate(model.named_parameters()):
            logging.info('{}: {}'.format(name, list(param.size())))
        parameter_num = sum([param.numel() for param in model.parameters()])
        logging.info('\nTotal number of parameters: {}\n'.format(parameter_num))

    '''
    3 Data generator
    '''
    if args.mode == 'train' or args.mode == 'test':
        hdf5_dir = os.path.join(args.feature_dir, args.feature_type,
                                hdf5_folder_name, args.audio_type + '_dev')
        data_generator = DevDataGenerator(
            args=args,
            hdf5_dir=hdf5_dir,
            logging=logging
        )
    elif args.mode == 'infer_eval':
        hdf5_dir = os.path.join(args.feature_dir, args.feature_type,
                                hdf5_folder_name, args.audio_type + '_eval')
        data_generator = EvalDataGenerator(
            args=args,
            hdf5_dir=hdf5_dir,
            logging=logging
        )
    '''
    4. Train, test and evaluation
    '''
    if args.mode == 'train':
        train(args, data_generator, model, optimizer, initial_epoch, logging)
    elif args.mode == 'test':
        test(args, data_generator, logging)
    elif args.mode == 'infer_eval':
        infer_eval(args, data_generator, logging)
