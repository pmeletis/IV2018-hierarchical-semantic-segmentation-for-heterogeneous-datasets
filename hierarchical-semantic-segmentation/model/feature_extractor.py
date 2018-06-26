"""User-defined feature extractor for dense semantic segmentation model.
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import resnet_utils
resnet_arg_scope = resnet_utils.resnet_arg_scope

def feature_extractor(mode, features, labels, config, params):
  """Fully Convolutional feature extractor for Semantic Segmentation.

  This function returns a feature extractor.
  First, the base feature extractor is created, which consists of a
  predefined network that is parameterized for the problem of SS.
  Then, an optional extension to the feature extractor is created
  (in series with the base) to deal the with feature dimensions and
  the receptive field of the feature representation specialized to SS.
  """
  # delete unused arguments from local namescope
  del labels, config

  with tf.variable_scope('feature_extractor'):
    # resnet base feature extractor scope arguments
    resnet_scope_args = {}
    if mode == tf.estimator.ModeKeys.TRAIN:
      resnet_scope_args.update(weight_decay=params.regularization_weight,
                               batch_norm_decay=params.batch_norm_decay)
    # build base of feature extractor
    with tf.variable_scope('base'), (
        slim.arg_scope(resnet_arg_scope(**resnet_scope_args))):
      # when num_classes=None no logits layer is created,
      #   when global_pool=False model is used for dense output
      fe, end_points = resnet_v1.resnet_v1_50(
          features,
          num_classes=None,
          is_training=params.batch_norm_istraining,
          global_pool=False,
          output_stride=params.stride_feature_extractor)

    # build extension to feature extractor
    #   decrease feature dimensions and increase field of view of
    #   feature extractor in a memory and computational efficient way
    #   hf/sfe x wf/sfe x 2048 8/32 (??) -->
    #   hf/sfe x wf/sfe x projection_dims 8/32 -->
    #   hf/sfe x wf/sfe x projection_dims 8/XX
    # TODO: add to end_points the outputs of next layers
    with tf.variable_scope('extension'):
      # WARNING: this scope assumes that slim.conv2d uses slim.batch_norm
      #   for the batch normalization, which holds at least up to TF v1.4
      with slim.arg_scope([slim.batch_norm], is_training=params.batch_norm_istraining), (
          slim.arg_scope(resnet_arg_scope(**resnet_scope_args))):
        if params.feature_dims_decreased > 0:
          fe = slim.conv2d(fe,
                           num_outputs=params.feature_dims_decreased,
                           kernel_size=1,
                           scope='decrease_fdims')
        if params.fov_expansion_kernel_rate > 0 and params.fov_expansion_kernel_size > 0:
          fe = slim.conv2d(fe,
                           num_outputs=fe.shape[-1],
                           kernel_size=params.fov_expansion_kernel_size,
                           rate=params.fov_expansion_kernel_rate,
                           scope='increase_fov')

  return fe, end_points, resnet_scope_args
