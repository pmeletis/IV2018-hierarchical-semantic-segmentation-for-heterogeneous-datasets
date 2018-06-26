"""User-defined dense semantic segmentation model.
"""

import functools
from contextlib import contextmanager
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import resnet_utils
resnet_arg_scope = resnet_utils.resnet_arg_scope
from model.feature_extractor import feature_extractor
from utils.utils import almost_equal

def model(mode, features, labels, config, params):

  # build the feature extractor
  # features: Nb x hf x wf x feature_dims_decreased(e.g. 256)
  features, end_points, fe_scope_args = feature_extractor(mode, features, labels, config, params)

  # level-specific sub-networks
  with tf.variable_scope('subnetworks'), \
      slim.arg_scope(resnet_arg_scope(**fe_scope_args)), \
      slim.arg_scope([slim.batch_norm], is_training=params.batch_norm_istraining):

    def _conv2d(features):
      return slim.conv2d(features, num_outputs=features.shape[-1].value, kernel_size=1)

    def _bottleneck(features, scope):
      return resnet_v1.bottleneck(features,
                                  features.shape[-1].value,
                                  features.shape[-1].value,
                                  1,
                                  scope=scope)

    # l1 features for classifying into 55+1 classes
    # l1_features = _conv2d(features)
    l1_features = _bottleneck(features, 'l1_features')
    # l2 features for classifying driveable, rider and traffic sign children into [10+1, 3+1, 2+1] classes
    l2_features = [_bottleneck(features, f"l2_features_driveable"),
                   _bottleneck(features, f"l2_features_rider"),
                   _bottleneck(features, f"l2_features_traffic_sign")]
    # l3 features for classifying traffic sign front children into [43+1] classes
    l3_features = _bottleneck(features, f"l3_features_traffic_sign_front")

  ## create logits, probabilities and top-1 decisions
  ##   First the logits are created and then upsampled for memory efficiency.
  with tf.variable_scope(
      'softmax_classifiers',
      initializer=slim.initializers.variance_scaling_initializer(),
      regularizer=slim.regularizers.l2_regularizer(params.regularization_weight)):
    def _conv2d(features, num_outputs, scope):
      return slim.conv2d(features,
                         num_outputs=num_outputs,
                         kernel_size=1,
                         activation_fn=None,
                         scope=scope)
    # l1 classes (animal, construction, human, marking, nature, object, void, unlabeled)
    l1_logits = [_conv2d(l1_features, 54, 'l1_logits')]
    l2_logits = [_conv2d(l2_features[0], 11, f"l2_logits_driveable"),
                 _conv2d(l2_features[1], 4, f"l2_logits_rider"),
                 _conv2d(l2_features[2], 3, f"l2_logits_traffic_sign")]
    l3_logits = [_conv2d(l3_features, 44, f"l3_features_traffic_sign_front")]

    # print('debug:logits:', l1_logits.op.name, l1_logits.shape)
    l1_logits = [_create_upsampler(l1_logits[0], params, scope='l1_upsampling')]
    # print('debug:upsampled logits:', l1_logits.op.name, l1_logits.shape)
    l2_logits = [_create_upsampler(l2_logits[0], params, scope=f"l2_upsampling_driveable"),
                 _create_upsampler(l2_logits[1], params, scope=f"l2_upsampling_rider"),
                 _create_upsampler(l2_logits[2], params, scope=f"l2_upsampling_traffic_sign")]
    l3_logits = [_create_upsampler(l3_logits[0], params, scope=f"l3_features_traffic_sign_front")]
    # during training probs and decs are used only for summaries
    # (not by train op) and thus not need to be computed in every step (GPU)
    # WARNING: next branch needed because if tf.device(None) is used,
    #   outer device allocation is not possible
    # def _device_cm(mode):
    #   # returns the appropriate device context manager according to mode
    #   @contextmanager
    #   def _useless_cm():
    #     yield
    #   cm = tf.device('/cpu:0') if mode == tf.estimator.ModeKeys.TRAIN else _useless_cm
    #   print(cm)
    #   with cm as sc:
    #     print(sc)
    #     return sc
    if mode == tf.estimator.ModeKeys.TRAIN:
      with tf.device('/cpu:0'): #_device_cm(mode):
        l1_probs = [tf.nn.softmax(l1_logits[0], name='l1_probabilities')]
        l1_decs = [tf.argmax(l1_probs[0], axis=3, output_type=tf.int32, name='l1_decisions')]
        l2_probs = [tf.nn.softmax(l2_logits[0], name=f"l2_probabilities_driveable"),
                    tf.nn.softmax(l2_logits[1], name=f"l2_probabilities_rider"),
                    tf.nn.softmax(l2_logits[2], name=f"l2_probabilities_traffic_sign")]
        l2_decs = [tf.argmax(l2_probs[0], axis=3, output_type=tf.int32, name=f"l2_decisions_driveable"),
                   tf.argmax(l2_probs[1], axis=3, output_type=tf.int32, name=f"l2_decisions_rider"),
                   tf.argmax(l2_probs[2], axis=3, output_type=tf.int32, name=f"l2_decisions_traffic_sign")]
        l3_probs = [tf.nn.softmax(l3_logits[0], name=f"l3_probabilities_traffic_sign_front")]
        l3_decs = [tf.argmax(l3_probs[0], axis=3, output_type=tf.int32, name=f"l3_traffic_sign_front")]
    else:
      l1_probs = [tf.nn.softmax(l1_logits[0], name='l1_probabilities')]
      l1_decs = [tf.argmax(l1_probs[0], axis=3, output_type=tf.int32, name='l1_decisions')]
      l2_probs = [tf.nn.softmax(l2_logits[0], name=f"l2_probabilities_driveable"),
                  tf.nn.softmax(l2_logits[1], name=f"l2_probabilities_rider"),
                  tf.nn.softmax(l2_logits[2], name=f"l2_probabilities_traffic_sign")]
      l2_decs = [tf.argmax(l2_probs[0], axis=3, output_type=tf.int32, name=f"l2_decisions_driveable"),
                  tf.argmax(l2_probs[1], axis=3, output_type=tf.int32, name=f"l2_decisions_rider"),
                  tf.argmax(l2_probs[2], axis=3, output_type=tf.int32, name=f"l2_decisions_traffic_sign")]
      l3_probs = [tf.nn.softmax(l3_logits[0], name=f"l3_probabilities_traffic_sign_front")]
      l3_decs = [tf.argmax(l3_probs[0], axis=3, output_type=tf.int32, name=f"l3_traffic_sign_front")]

  ## model outputs groupped as predictions of the Estimator
  # WARNING: 'decisions' key is used internally so it must exist for now..
  predictions = {
      'logits': [l1_logits, l2_logits, l3_logits],
      'probabilities': [l1_probs, l2_probs, l3_probs],
      'decisions': [l1_decs, l2_decs, l3_decs],
      }

  return features, end_points, predictions

def _create_upsampler(bottom, params, scope='upsampling'):
  # upsample bottom depthwise to reach feature extractor output dimensions
  # bottom: Nb x hf//sfe x hf//sfe x C
  # upsampled: Nb x hf x hf x C
  # TODO: Does upsampler needs regularization??
  # TODO: align_corners=params.enable_xla: XLA implements only align_corners=True for now,
  # change it when XLA implements all

  C = bottom.shape[-1]
  spat_dims = np.array(bottom.shape.as_list()[1:3])
  hf, wf = params.height_feature_extractor, params.width_feature_extractor
  # WARNING: Resized images will be distorted if their original aspect ratio is not the same as size (only in the case of bilinear resizing)
  if params.upsampling_method != 'no':
    assert almost_equal(spat_dims[0]/spat_dims[1], hf/wf, 10**-1), (
        f"Resized images will be distorted if their original aspect ratio is "
        f"not the same as size: {spat_dims[0],spat_dims[1]}, {hf,wf}.")
  with tf.variable_scope(scope):
    if params.upsampling_method == 'no':
      upsampled = bottom
    elif params.upsampling_method == 'bilinear':
      upsampled = tf.image.resize_images(bottom, [hf, wf], align_corners=params.enable_xla)
    elif params.upsampling_method == 'hybrid':
      # composite1: deconv upsample twice and then resize
      assert params.stride_feature_extractor in (4, 8, 16), 'stride_feature_extractor must be 4, 8 or 16.'
      upsampled = slim.conv2d_transpose(inputs=bottom,
                                        num_outputs=C,
                                        kernel_size=2*2,
                                        stride=2,
                                        padding='SAME',
                                        activation_fn=None,
                                        weights_initializer=slim.variance_scaling_initializer(),
                                        weights_regularizer=slim.l2_regularizer(params.regularization_weight))
      upsampled = tf.image.resize_images(upsampled, [hf, wf], align_corners=params.enable_xla)
    else:
      raise ValueError('No such upsampling method.')

  return upsampled
