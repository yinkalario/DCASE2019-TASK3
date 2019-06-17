#!/bin/bash

DEV_MODELS_PATH='dev_submission'
EVAL_MODELS_PATH='eval_submission'

DEV_OUTPUT_PATH='stacked_dev'
EVAL_OUTPUT_PATH='stacked_dev'

python meta_features.py "$DEV_MODELS_PATH" training_dev.h5
python meta_features.py "$EVAL_MODELS_PATH" training_eval.h5

python predict_stack.py training_dev.h5 "$DEV_MODELS_PATH/gt" "$DEV_OUTPUT_PATH"
python predict_stack.py training_dev.h5 --test_path training_eval.h5 "$DEV_MODELS_PATH/gt" "$EVAL_OUTPUT_PATH"
