#!/bin/bash

# Data directory
DATASET_DIR='/vol/vssp/AP_datasets/audio/dcase2019/task3/dataset_root/'

# Feature directory
FEATURE_DIR='/vol/vssp/msos/YinC/workspace/Dataset_Features/DCASE2019/task3/'

# Workspace
WORKSPACE='/vol/vssp/msos/YinC/workspace/DCASE2019/task3/mycode_v21/'
cd $WORKSPACE

########### Hyper-parameters ###########
FEATURE_TYPE='logmelgccintensity'  # 'logmel' | 'logmelgcc' | 'logmelintensity' | 'logmelgccintensity'
AUDIO_TYPE='foa&mic'                # 'mic' | 'foa' | 'foa&mic'

############ Extract Features ############
# dev
python utils/feature_extractor.py --dataset_dir=$DATASET_DIR --feature_dir=$FEATURE_DIR --feature_type=$FEATURE_TYPE --data_type='dev' --audio_type=$AUDIO_TYPE

# eval
python utils/feature_extractor.py --dataset_dir=$DATASET_DIR --feature_dir=$FEATURE_DIR --feature_type=$FEATURE_TYPE --data_type='eval' --audio_type=$AUDIO_TYPE
