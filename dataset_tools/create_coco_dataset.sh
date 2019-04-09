#!/usr/bin/env bash

TRAIN_IMAGE_DIR="/aidata/dataset/image_dataset/coco/train2014/"
VAL_IMAGE_DIR="/aidata/dataset/image_dataset/coco/val2014/"
TRAIN_ANNOTATIONS_FILE="/aidata/dataset/image_dataset/coco/annotations/instances_train2014.json"
VAL_ANNOTATIONS_FILE="/aidata/dataset/image_dataset/coco/annotations/instances_val2014.json"
OUTPUT_DIR="/aidata/huangyu/tmp/coco/"

python dataset_tools/create_coco_tf_record.py --logtostderr \
  --train_image_dir="${TRAIN_IMAGE_DIR}" \
  --val_image_dir="${VAL_IMAGE_DIR}" \
  --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
  --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
  --output_dir="${OUTPUT_DIR}"
