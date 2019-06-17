#!/bin/bash

# Data directory
DATASET_DIR='/vol/vssp/AP_datasets/audio/dcase2019/task3/dataset_root/'

# Feature directory
FEATURE_DIR='/vol/vssp/msos/YinC/workspace/Dataset_Features/DCASE2019/task3/'

# Workspace
WORKSPACE='/vol/vssp/msos/YinC/workspace/DCASE2019/task3/'
cd $WORKSPACE

########### Hyper-parameters ###########
FEATURE_TYPE='logmelgccintensity'  # 'logmel' | 'logmelgcc' | 'logmelintensity' | 'logmelgccintensity'
AUDIO_TYPE='foa&mic'                # 'mic' | 'foa' | 'foa&mic'
FOLD=-1

# Chunk length
CHUNKLEN=5

# Model
MODEL_SED='CRNN9_logmelgccintensity'              # 'CRNN11' | 'CRNN9' | 'Gated_CRNN9' | 'CRNN9_logmelgccintensity' | 'CRNN11_logmelgccintensity'
MODEL_DOA='pretrained_CRNN8_logmelgccintensity'   # 'pretrained_CRNN10' | 'pretrained_CRNN8' | 'pretrained_Gated_CRNN8' | 'pretrained_CRNN8_logmelgccintensity' | 'pretrained_CRNN10_logmelgccintensity'

# Data augmentation
DATA_AUG='None'                 # 'None' | 'mixup' | 'specaug' | 'mixup&specaug'

# Name of the trial
NAME='BS32_5s'

# seed
SEED=30250

# GPU number
GPU_ID=0

############ Development Evaluation ############
# test SED first
FUSION='True'                  # Ensemble or not: 'True' | 'False'
TASK_TYPE='sed_only'            # 'sed_only' | 'doa_only' | 'two_staged_eval' | 'seld'
for EPOCH_NUM in {38..40..2}
    do  
    echo $'\nEpoch numbers: '$EPOCH_NUM
    CUDA_VISIBLE_DEVICES=$GPU_ID python ${WORKSPACE}main/main.py infer_eval --workspace=$WORKSPACE --feature_dir=$FEATURE_DIR --feature_type=$FEATURE_TYPE \
    --audio_type=$AUDIO_TYPE --task_type=$TASK_TYPE --model_sed=$MODEL_SED --model_doa=$MODEL_DOA --fold=$FOLD --epoch_num=$EPOCH_NUM \
    --data_aug=$DATA_AUG --seed=$SEED --name=$NAME --fusion=$FUSION
done

# ensemble sed on different iterations and write out probabilities
python ${WORKSPACE}main/ensemble.py iters_ensemble --workspace=$WORKSPACE --feature_type=$FEATURE_TYPE --audio_type=$AUDIO_TYPE --task_type=$TASK_TYPE \
--model_sed=$MODEL_SED --model_doa=$MODEL_DOA --data_aug=$DATA_AUG --seed=$SEED --name=$NAME

# ensemble sed on different models and write out probabilities
python ${WORKSPACE}main/ensemble.py models_ensemble --workspace=$WORKSPACE --feature_type=$FEATURE_TYPE --audio_type=$AUDIO_TYPE --task_type=$TASK_TYPE \
--model_sed=$MODEL_SED --model_doa=$MODEL_DOA --data_aug=$DATA_AUG --seed=$SEED --name=$NAME

# threshold the probabilities and write out submissions to 'sed_test_fusioned' folder
THRESHOLD=0.5

# test DOA
FUSION='True'                  # Ensemble or not: 'True' | 'False'
TASK_TYPE='two_staged_eval'    # 'sed_only' | 'doa_only' | 'two_staged_eval' | 'seld'
for EPOCH_NUM in {78..80..2}
    do  
    echo $'\nEpoch numbers: '$EPOCH_NUM
    CUDA_VISIBLE_DEVICES=$GPU_ID python ${WORKSPACE}main/main.py infer_eval --workspace=$WORKSPACE --feature_dir=$FEATURE_DIR --feature_type=$FEATURE_TYPE \
    --audio_type=$AUDIO_TYPE --task_type=$TASK_TYPE --model_sed=$MODEL_SED --model_doa=$MODEL_DOA --fold=$FOLD --epoch_num=$EPOCH_NUM \
    --data_aug=$DATA_AUG --seed=$SEED --name=$NAME --fusion=$FUSION
done

# ensemble doa
python ${WORKSPACE}main/ensemble.py iters_ensemble --workspace=$WORKSPACE --feature_type=$FEATURE_TYPE --audio_type=$AUDIO_TYPE --task_type=$TASK_TYPE \
--model_sed=$MODEL_SED --model_doa=$MODEL_DOA --data_aug=$DATA_AUG --seed=$SEED --name=$NAME

# ensemble sed on different models and write out probabilities
python ${WORKSPACE}main/ensemble.py models_ensemble --workspace=$WORKSPACE --feature_type=$FEATURE_TYPE --audio_type=$AUDIO_TYPE --task_type=$TASK_TYPE \
--model_sed=$MODEL_SED --model_doa=$MODEL_DOA --data_aug=$DATA_AUG --seed=$SEED --name=$NAME

# threshold the probabilities and write out submissions to 'sed_test_fusioned' folder
THRESHOLD=0.5
python ${WORKSPACE}main/ensemble.py threshold_models_ensemble --workspace=$WORKSPACE --feature_type=$FEATURE_TYPE --audio_type=$AUDIO_TYPE --task_type=$TASK_TYPE \
--model_sed=$MODEL_SED --model_doa=$MODEL_DOA --data_aug=$DATA_AUG --seed=$SEED --name=$NAME --threshold=$THRESHOLD
