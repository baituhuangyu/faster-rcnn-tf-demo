# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    python object_detection/dataset_tools/create_pascal_tf_record.py \
        --data_dir=/home/user/VOCdevkit \
        --year=VOC2012 \
        --output_path=/home/user/pascal.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os

from lxml import etree
import PIL.Image
import tensorflow as tf
import contextlib2
import numpy as np

from utils import dataset_util
from utils import label_map_util

from dataset_tools import tf_record_creation_util
from utils.visualization_utils import draw_bounding_boxes_on_image


flags = tf.app.flags

flags.DEFINE_string('base_data_dir', '', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                    'merged set.')

flags.DEFINE_string('train_sub_image_dir', '',
                    '(Relative) path to image directory.')
flags.DEFINE_string('val_sub_image_dir', 'Annotations',
                    '(Relative) path to image directory.')
flags.DEFINE_string('train_sub_annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('val_sub_annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('output_dir', '', 'Path to output TFRecord')

flags.DEFINE_string('label_map_path', 'data/pascal_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances')
FLAGS = flags.FLAGS

SETS = ['train', 'val', 'trainval', 'test']
YEARS = ['VOC2007', 'VOC2012', 'merged']


def dict_to_tf_example(data,
                       img_path,
                       label_map_dict,
                       ignore_difficult_instances=False,
                       image_subdirectory='JPEGImages'):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    dataset_directory: Path to root directory holding PASCAL dataset
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    image_subdirectory: String specifying subdirectory within the
      PASCAL dataset directory holding the actual image data.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """

  full_path = img_path
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  if 'object' in data:
    for obj in data['object']:
      difficult = bool(int(obj['difficult']))
      if ignore_difficult_instances and difficult:
        continue
      if int(obj['name']) == 3:
        continue

      difficult_obj.append(int(difficult))

      xmin.append(float(obj['bndbox']['xmin']) / width)
      ymin.append(float(obj['bndbox']['ymin']) / height)
      xmax.append(float(obj['bndbox']['xmax']) / width)
      ymax.append(float(obj['bndbox']['ymax']) / height)
      # classes_text.append(obj['name'].encode('utf8'))
      classes_text.append(label_map_dict[obj['name']].encode('utf8'))
      classes.append(int(obj['name']))
      truncated.append(int(obj['truncated']))
      poses.append(obj['pose'].encode('utf8'))

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }))
  return example


def _create_tf_record_from_coco_annotations(fs, output_path, num_shards=100):
    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, output_path, num_shards)

        label_map_dict = {
            "1": "1",
            "2": "2",
            "3": "others"
        }
        for idx, example in enumerate(fs):
            shard_idx = idx % num_shards
            image_path = example[0]
            annotation_file = example[1]
            if shard_idx == 0:
                logging.info('On image %d', idx)
            assert os.path.basename(image_path).replace(".jpg", "") == os.path.basename(annotation_file).replace(".xml", "")
            with tf.gfile.GFile(annotation_file, 'r') as fid:
                xml_str = fid.read()
            try:
                xml = etree.fromstring(xml_str)
            except Exception as ex:
                print(ex)
                xml_str = "\n".join(xml_str.split("\n")[1:])
                xml = etree.fromstring(xml_str)
            data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
            tf_example = dict_to_tf_example(data, image_path, label_map_dict, FLAGS.ignore_difficult_instances)
            output_tfrecords[shard_idx].write(tf_example.SerializeToString())


def _get_all_file_path(base_data_dir, train_sub_image_dir, val_sub_image_dir, train_sub_annotations_dir, val_sub_annotations_dir):
    train_flag = "train"
    val_flag = "val"

    # train image
    train_image_dir = os.path.join(base_data_dir, train_flag, train_sub_image_dir)
    train_image_names = sorted(os.listdir(train_image_dir))
    train_image_names = [_ for _ in train_image_names if _.endswith(".jpg")]
    train_image_path_list = [os.path.join(train_image_dir, _) for _ in train_image_names]

    # val image
    val_image_dir = os.path.join(base_data_dir, val_flag, val_sub_image_dir)
    val_image_names = sorted(os.listdir(val_image_dir))
    val_image_names = [_ for _ in val_image_names if _.endswith(".jpg")]
    val_image_path_list = [os.path.join(val_image_dir, _) for _ in val_image_names]

    # train annotations
    train_annotation_dir = os.path.join(base_data_dir, train_flag, train_sub_annotations_dir)
    train_annotation_name = sorted(os.listdir(train_annotation_dir))
    train_annotation_name = [_ for _ in train_annotation_name if _.endswith(".xml")]
    train_annotation_path_list = [os.path.join(train_annotation_dir, _) for _ in train_annotation_name]

    # val annotations
    val_annotation_dir = os.path.join(base_data_dir, val_flag, val_sub_annotations_dir)
    val_annotation_name = sorted(os.listdir(val_annotation_dir))
    val_annotation_name = [_ for _ in val_annotation_name if _.endswith(".xml")]
    val_annotation_path_list = [os.path.join(val_annotation_dir, _) for _ in val_annotation_name]

    # check
    assert len(train_image_path_list) == len(train_annotation_path_list)
    assert len(val_image_path_list) == len(val_annotation_path_list)

    return zip(train_image_path_list, train_annotation_path_list), zip(val_image_path_list, val_annotation_path_list)


def statistic():
    annotations_dir = "/Users/hy/Documents/coco/Annotations/"

    all_xml_fs = [os.path.join(annotations_dir, _) for _ in sorted(os.listdir(annotations_dir))]
    all_xml_fs = [_ for _ in all_xml_fs if _.endswith(".xml")]

    names = []
    pose = []
    truncated = []
    difficult = []

    width = []
    height = []
    depth = []

    xmin = []
    ymin = []
    xmax = []
    ymax = []

    for annotation_file in all_xml_fs:
        with tf.gfile.GFile(annotation_file, 'r') as fid:
            xml_str = fid.read()
        try:
            xml = etree.fromstring(xml_str)
        except Exception as ex:
            print(ex)
            xml_str = "\n".join(xml_str.split("\n")[1:])
            xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

        names.extend([_["name"] for _ in data['object']])
        pose.extend([_["pose"] for _ in data['object']])
        truncated.extend([_["truncated"] for _ in data['object']])
        difficult.extend([_["difficult"] for _ in data['object']])

        width.append(data["size"]["width"])
        height.append(data["size"]["height"])
        depth.append(data["size"]["depth"])

        xmin.append(min([float(_['bndbox']['xmin']) for _ in data['object']]))
        ymin.append(min([float(_['bndbox']['ymin']) for _ in data['object']]))
        xmax.append(max([float(_['bndbox']['xmax']) for _ in data['object']]))
        ymax.append(max([float(_['bndbox']['ymax']) for _ in data['object']]))

    print(set(names))
    # print(set(pose))
    # print(set(truncated))
    # print(set(difficult))
    #
    # print(set(width))
    # print(set(height))
    # print(set(depth))
    #
    # print(min(xmin))
    # print(min(ymin))
    # print(max(xmax))
    # print(max(ymax))


def xxx(fs):
    for idx, example in enumerate(fs):
        print("idx", idx)
        if idx < 10:
            continue

        image_path = example[0]
        annotation_file = example[1]
        print("image_path", image_path)
        print("annotation_file", annotation_file)
        assert os.path.basename(image_path).replace(".jpg", "") == os.path.basename(annotation_file).replace(".xml", "")
        with tf.gfile.GFile(annotation_file, 'r') as fid:
            xml_str = fid.read()
        try:
            xml = etree.fromstring(xml_str)
        except Exception as ex:
            print(ex)
            xml_str = "\n".join(xml_str.split("\n")[1:])
            xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

        full_path = image_path
        with tf.gfile.GFile(full_path, 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = PIL.Image.open(encoded_jpg_io)

        width = int(data['size']['width'])
        height = int(data['size']['height'])

        print("width", width)
        print("height", height)

        xmin = []
        ymin = []
        xmax = []
        ymax = []
        if 'object' in data:
            for obj in data['object']:
                if int(obj['name']) != 2:
                    continue

                obj_xmin = float(obj['bndbox']['xmin'])
                obj_ymin = float(obj['bndbox']['ymin'])
                obj_xmax = float(obj['bndbox']['xmax'])
                obj_ymax = float(obj['bndbox']['ymax'])

                # assert width > obj_xmin > 0
                # assert height > obj_ymin > 0
                # assert width > obj_xmax > 0
                # assert height > obj_ymax > 0
                #
                # assert obj_xmin < obj_xmax
                # assert obj_ymin < obj_ymax

                xmin.append(obj_xmin / width)
                ymin.append(obj_ymin / height)
                xmax.append(obj_xmax / width)
                ymax.append(obj_ymax / height)

                # xmin.append(obj_xmin)
                # ymin.append(obj_ymin)
                # xmax.append(obj_xmax)
                # ymax.append(obj_ymax)

        bboxes = np.array([ymin, xmin, ymax, xmax]).transpose([1, 0])

        image = image.convert("RGB")
        draw_bounding_boxes_on_image(image, bboxes, color="red", thickness=4)

        PIL.Image.Image.show(image)

def main(_):

  base_data_dir = FLAGS.base_data_dir
  set_flag = FLAGS.set
  train_sub_image_dir = FLAGS.train_sub_image_dir
  val_sub_image_dir = FLAGS.val_sub_image_dir
  train_sub_annotations_dir = FLAGS.train_sub_annotations_dir
  val_sub_annotations_dir = FLAGS.val_sub_annotations_dir
  output_dir = FLAGS.output_dir

  train_fs, val_fs = _get_all_file_path(base_data_dir, train_sub_image_dir, val_sub_image_dir,
                                        train_sub_annotations_dir, val_sub_annotations_dir)

  _create_tf_record_from_coco_annotations(train_fs, os.path.join(output_dir, "train"), 100)
  _create_tf_record_from_coco_annotations(val_fs, os.path.join(output_dir, "val"), 100)

  # statistic()
  # xxx(train_fs)


if __name__ == '__main__':
  tf.app.run()
