import os

import tensorflow as tf


def train_init(config, params, scope='init'):
  # different situations for initialization:
  #   1) initialize from init_ckpt_path (log_dir has to be empty from checkpoints)
  #   2) continue training from log_dir

  del config

  with tf.name_scope(scope), tf.device('/cpu:0'):

    # one of those must given (xor)
    assert bool(params.init_ckpt_path) != bool(tf.train.latest_checkpoint(params.log_dir)), (
        'If init_ckpt_path is given log_dir has to be empty of checkpoints, '
        'if log_dir is given training continuous from latest checkpoint and '
        'init_ckpt_path has to be empty.')

    ## initialize from checkpoint, e.g. trained on ImageNet
    if params.init_ckpt_path:
      # the best we can do is to initialize from the checkpoint as much variables as possible
      # so we find the mapping from checkpoint names to model names
      # assumes names in model are extended with a prefix from names in checkpoint
      # e.g.
      # in checkpoint: resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights
      # in model: feature_extractor/base/resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights

      # list of (name, shape) of checkpoint variables
      ckpt_vars = tf.train.list_variables(params.init_ckpt_path)
      # list of tf.Variable of model variables
      global_vars = tf.global_variables()

      # checkpoint variable name --> model variable mappings
      # TODO: exclude variables in a better way, still the parts below may be included in a
      # useful variable, e.g. use moving_average_variables and variables from model.py
      exclude = ['global_step', 'train_ops', 'ExponentialMovingAverage',
                 'Momentum', 'classifier', 'extension']
      var_dict = dict()
      for gv in global_vars:
        for cvn, cvs in ckpt_vars:
          for exc in exclude:
            if exc in gv.op.name:
              break
          else:
            if cvn in gv.op.name and tf.TensorShape(cvs).is_compatible_with(gv.shape):
              var_dict[cvn] = gv

      extra_vars_to_init = set(global_vars).difference(set(var_dict.values()))

      # for k, v in var_dict.items():
      #   print(k, v)
      # print('gv', len(global_vars), 'cv', len(ckpt_vars), 'inter', len(var_dict))

      init_op, init_feed_dict = tf.contrib.framework.assign_from_checkpoint(
          params.init_ckpt_path,
          var_dict,
          ignore_missing_vars=False)
      extra_init_op = tf.variables_initializer(extra_vars_to_init)

      return tf.group(init_op, extra_init_op), init_feed_dict

    else:

      return None, None
