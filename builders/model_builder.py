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

"""A function to build a DetectionModel from configuration."""

import functools

from builders import anchor_generator_builder
from builders import box_coder_builder
from builders import box_predictor_builder
from builders import hyperparams_builder
from builders import image_resizer_builder
from builders import losses_builder
from builders import matcher_builder
from builders import post_processing_builder
from builders import region_similarity_calculator_builder as sim_calc
from core import balanced_positive_negative_sampler as sampler
from core import post_processing
from core import target_assigner
from meta_architectures import faster_rcnn_meta_arch
from models import faster_rcnn_inception_resnet_v2_feature_extractor as frcnn_inc_res
from models import faster_rcnn_nas_feature_extractor as frcnn_nas
from models import faster_rcnn_pnas_feature_extractor as frcnn_pnas
from predictors import rfcn_box_predictor
from predictors.heads import mask_head
from protos import model_pb2
from utils import ops


# A map of names to Faster R-CNN feature extractors.
FASTER_RCNN_FEATURE_EXTRACTOR_CLASS_MAP = {
    'faster_rcnn_nas':
    frcnn_nas.FasterRCNNNASFeatureExtractor,
    'faster_rcnn_pnas':
    frcnn_pnas.FasterRCNNPNASFeatureExtractor,
    'faster_rcnn_inception_resnet_v2':
    frcnn_inc_res.FasterRCNNInceptionResnetV2FeatureExtractor,
    # 'faster_rcnn_inception_v2':
    # frcnn_inc_v2.FasterRCNNInceptionV2FeatureExtractor,
    # 'faster_rcnn_resnet50':
    # frcnn_resnet_v1.FasterRCNNResnet50FeatureExtractor,
    # 'faster_rcnn_resnet101':
    # frcnn_resnet_v1.FasterRCNNResnet101FeatureExtractor,
    # 'faster_rcnn_resnet152':
    # frcnn_resnet_v1.FasterRCNNResnet152FeatureExtractor,
}


def build(model_config, is_training, add_summaries=True):
  """Builds a DetectionModel based on the model config.

  Args:
    model_config: A model.proto object containing the config for the desired
      DetectionModel.
    is_training: True if this model is being built for training purposes.
    add_summaries: Whether to add tensorflow summaries in the model graph.
  Returns:
    DetectionModel based on the config.

  Raises:
    ValueError: On invalid meta architecture or model.
  """
  if not isinstance(model_config, model_pb2.DetectionModel):
    raise ValueError('model_config not of type model_pb2.DetectionModel.')
  meta_architecture = model_config.WhichOneof('model')
  # if meta_architecture == 'ssd':
  #   return _build_ssd_model(model_config.ssd, is_training, add_summaries)
  if meta_architecture == 'faster_rcnn':
    return _build_faster_rcnn_model(model_config.faster_rcnn, is_training,
                                    add_summaries)
  raise ValueError('Unknown meta architecture: {}'.format(meta_architecture))


def _build_faster_rcnn_feature_extractor(
    feature_extractor_config, is_training, reuse_weights=None,
    inplace_batchnorm_update=False):
  """Builds a faster_rcnn_meta_arch.FasterRCNNFeatureExtractor based on config.

  Args:
    feature_extractor_config: A FasterRcnnFeatureExtractor proto config from
      faster_rcnn.proto.
    is_training: True if this feature extractor is being built for training.
    reuse_weights: if the feature extractor should reuse weights.
    inplace_batchnorm_update: Whether to update batch_norm inplace during
      training. This is required for batch norm to work correctly on TPUs. When
      this is false, user must add a control dependency on
      tf.GraphKeys.UPDATE_OPS for train/loss op in order to update the batch
      norm moving average parameters.

  Returns:
    faster_rcnn_meta_arch.FasterRCNNFeatureExtractor based on config.

  Raises:
    ValueError: On invalid feature extractor type.
  """
  if inplace_batchnorm_update:
    raise ValueError('inplace batchnorm updates not supported.')
  feature_type = feature_extractor_config.type
  first_stage_features_stride = (
      feature_extractor_config.first_stage_features_stride)
  batch_norm_trainable = feature_extractor_config.batch_norm_trainable

  if feature_type not in FASTER_RCNN_FEATURE_EXTRACTOR_CLASS_MAP:
    raise ValueError('Unknown Faster R-CNN feature_extractor: {}'.format(
        feature_type))
  feature_extractor_class = FASTER_RCNN_FEATURE_EXTRACTOR_CLASS_MAP[
      feature_type]
  return feature_extractor_class(
      is_training, first_stage_features_stride,
      batch_norm_trainable, reuse_weights)


def _build_faster_rcnn_model(frcnn_config, is_training, add_summaries):
  """Builds a Faster R-CNN or R-FCN detection model based on the model config.

  Builds R-FCN model if the second_stage_box_predictor in the config is of type
  `rfcn_box_predictor` else builds a Faster R-CNN model.

  Args:
    frcnn_config: A faster_rcnn.proto object containing the config for the
      desired FasterRCNNMetaArch or RFCNMetaArch.
    is_training: True if this model is being built for training purposes.
    add_summaries: Whether to add tf summaries in the model.

  Returns:
    FasterRCNNMetaArch based on the config.

  Raises:
    ValueError: If frcnn_config.type is not recognized (i.e. not registered in
      model_class_map).
  """
  num_classes = frcnn_config.num_classes
  image_resizer_fn = image_resizer_builder.build(frcnn_config.image_resizer)

  feature_extractor = _build_faster_rcnn_feature_extractor(
      frcnn_config.feature_extractor, is_training,
      inplace_batchnorm_update=frcnn_config.inplace_batchnorm_update)

  number_of_stages = frcnn_config.number_of_stages
  first_stage_anchor_generator = anchor_generator_builder.build(
      frcnn_config.first_stage_anchor_generator)

  first_stage_target_assigner = target_assigner.create_target_assigner(
      'FasterRCNN',
      'proposal',
      use_matmul_gather=frcnn_config.use_matmul_gather_in_matcher)
  first_stage_atrous_rate = frcnn_config.first_stage_atrous_rate
  first_stage_box_predictor_arg_scope_fn = hyperparams_builder.build(
      frcnn_config.first_stage_box_predictor_conv_hyperparams, is_training)
  first_stage_box_predictor_kernel_size = (
      frcnn_config.first_stage_box_predictor_kernel_size)
  first_stage_box_predictor_depth = frcnn_config.first_stage_box_predictor_depth
  first_stage_minibatch_size = frcnn_config.first_stage_minibatch_size
  use_static_shapes = frcnn_config.use_static_shapes and (
      frcnn_config.use_static_shapes_for_eval or is_training)
  first_stage_sampler = sampler.BalancedPositiveNegativeSampler(
      positive_fraction=frcnn_config.first_stage_positive_balance_fraction,
      is_static=(frcnn_config.use_static_balanced_label_sampler and
                 use_static_shapes))
  first_stage_max_proposals = frcnn_config.first_stage_max_proposals
  if (frcnn_config.first_stage_nms_iou_threshold < 0 or
      frcnn_config.first_stage_nms_iou_threshold > 1.0):
    raise ValueError('iou_threshold not in [0, 1.0].')
  if (is_training and frcnn_config.second_stage_batch_size >
      first_stage_max_proposals):
    raise ValueError('second_stage_batch_size should be no greater than '
                     'first_stage_max_proposals.')
  first_stage_non_max_suppression_fn = functools.partial(
      post_processing.batch_multiclass_non_max_suppression,
      score_thresh=frcnn_config.first_stage_nms_score_threshold,
      iou_thresh=frcnn_config.first_stage_nms_iou_threshold,
      max_size_per_class=frcnn_config.first_stage_max_proposals,
      max_total_size=frcnn_config.first_stage_max_proposals,
      use_static_shapes=use_static_shapes)
  first_stage_loc_loss_weight = (
      frcnn_config.first_stage_localization_loss_weight)
  first_stage_obj_loss_weight = frcnn_config.first_stage_objectness_loss_weight

  initial_crop_size = frcnn_config.initial_crop_size
  maxpool_kernel_size = frcnn_config.maxpool_kernel_size
  maxpool_stride = frcnn_config.maxpool_stride

  second_stage_target_assigner = target_assigner.create_target_assigner(
      'FasterRCNN',
      'detection',
      use_matmul_gather=frcnn_config.use_matmul_gather_in_matcher)
  second_stage_box_predictor = box_predictor_builder.build(
      hyperparams_builder.build,
      frcnn_config.second_stage_box_predictor,
      is_training=is_training,
      num_classes=num_classes)
  second_stage_batch_size = frcnn_config.second_stage_batch_size
  second_stage_sampler = sampler.BalancedPositiveNegativeSampler(
      positive_fraction=frcnn_config.second_stage_balance_fraction,
      is_static=(frcnn_config.use_static_balanced_label_sampler and
                 use_static_shapes))
  (second_stage_non_max_suppression_fn, second_stage_score_conversion_fn
  ) = post_processing_builder.build(frcnn_config.second_stage_post_processing)
  second_stage_localization_loss_weight = (
      frcnn_config.second_stage_localization_loss_weight)
  second_stage_classification_loss = (
      losses_builder.build_faster_rcnn_classification_loss(
          frcnn_config.second_stage_classification_loss))
  second_stage_classification_loss_weight = (
      frcnn_config.second_stage_classification_loss_weight)
  second_stage_mask_prediction_loss_weight = (
      frcnn_config.second_stage_mask_prediction_loss_weight)

  hard_example_miner = None
  if frcnn_config.HasField('hard_example_miner'):
    hard_example_miner = losses_builder.build_hard_example_miner(
        frcnn_config.hard_example_miner,
        second_stage_classification_loss_weight,
        second_stage_localization_loss_weight)

  crop_and_resize_fn = (
      ops.matmul_crop_and_resize if frcnn_config.use_matmul_crop_and_resize
      else ops.native_crop_and_resize)
  clip_anchors_to_image = (
      frcnn_config.clip_anchors_to_image)

  common_kwargs = {
      'is_training': is_training,
      'num_classes': num_classes,
      'image_resizer_fn': image_resizer_fn,
      'feature_extractor': feature_extractor,
      'number_of_stages': number_of_stages,
      'first_stage_anchor_generator': first_stage_anchor_generator,
      'first_stage_target_assigner': first_stage_target_assigner,
      'first_stage_atrous_rate': first_stage_atrous_rate,
      'first_stage_box_predictor_arg_scope_fn':
      first_stage_box_predictor_arg_scope_fn,
      'first_stage_box_predictor_kernel_size':
      first_stage_box_predictor_kernel_size,
      'first_stage_box_predictor_depth': first_stage_box_predictor_depth,
      'first_stage_minibatch_size': first_stage_minibatch_size,
      'first_stage_sampler': first_stage_sampler,
      'first_stage_non_max_suppression_fn': first_stage_non_max_suppression_fn,
      'first_stage_max_proposals': first_stage_max_proposals,
      'first_stage_localization_loss_weight': first_stage_loc_loss_weight,
      'first_stage_objectness_loss_weight': first_stage_obj_loss_weight,
      'second_stage_target_assigner': second_stage_target_assigner,
      'second_stage_batch_size': second_stage_batch_size,
      'second_stage_sampler': second_stage_sampler,
      'second_stage_non_max_suppression_fn':
      second_stage_non_max_suppression_fn,
      'second_stage_score_conversion_fn': second_stage_score_conversion_fn,
      'second_stage_localization_loss_weight':
      second_stage_localization_loss_weight,
      'second_stage_classification_loss':
      second_stage_classification_loss,
      'second_stage_classification_loss_weight':
      second_stage_classification_loss_weight,
      'hard_example_miner': hard_example_miner,
      'add_summaries': add_summaries,
      'crop_and_resize_fn': crop_and_resize_fn,
      'clip_anchors_to_image': clip_anchors_to_image,
      'use_static_shapes': use_static_shapes,
      'resize_masks': frcnn_config.resize_masks
  }

  if isinstance(second_stage_box_predictor,
                rfcn_box_predictor.RfcnBoxPredictor):
      print("rfcn_meta_arch.RFCNMetaArch")
      pass
    # return rfcn_meta_arch.RFCNMetaArch(
    #     second_stage_rfcn_box_predictor=second_stage_box_predictor,
    #     **common_kwargs)
  else:
    return faster_rcnn_meta_arch.FasterRCNNMetaArch(
        initial_crop_size=initial_crop_size,
        maxpool_kernel_size=maxpool_kernel_size,
        maxpool_stride=maxpool_stride,
        second_stage_mask_rcnn_box_predictor=second_stage_box_predictor,
        second_stage_mask_prediction_loss_weight=(
            second_stage_mask_prediction_loss_weight),
        **common_kwargs)
