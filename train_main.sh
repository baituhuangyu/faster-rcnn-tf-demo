#!/usr/bin/env bash

# From the tensorflow/models/research/ directory
PIPELINE_CONFIG_PATH="/aidata/huangyu/ml_test/coco/sample/config/pipeline.config"
MODEL_DIR="/aidata/huangyu/tmp/coco/save_model"
NUM_TRAIN_STEPS=50000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr