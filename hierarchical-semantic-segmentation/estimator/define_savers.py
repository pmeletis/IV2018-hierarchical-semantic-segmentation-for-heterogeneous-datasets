import tensorflow as tf

def train_saver(config, params, scope='saver'):
  # saver intended for saving vars
  # WARNING: for now TF does not separate between saver and loaders: that is it
  # uses savers also to load a model, thus variables in exclude must be initialized manually
  if params.init_ckpt_path:
    with tf.name_scope(scope):
      # exclude = ['train_ops']
      exclude = []
      var_list = []
      for var in tf.global_variables():
        for exc in exclude:
          if exc.op.name in var.op.name:
            break
        else:
          var_list.append(var)

      # print('gv', len(tf.global_variables()), 'sv', len(var_list))

      saver = tf.train.Saver(var_list,
                             sharded=True,
                             max_to_keep=config.keep_checkpoint_max,
                             save_relative_paths=True)
  else:
    saver = None

  return saver

def evaluate_saver(config, params, scope='saver'):
  with tf.name_scope(scope):
    var_dict = dict()
    for mv in tf.model_variables():
      k = mv.op.name + ('/ExponentialMovingAverage' if params.restore_emas else '')
      var_dict[k] = mv

    # for now only global_step is in rest_vars
    rest_vars = set(tf.global_variables()).difference(set(var_dict.values()))
    for rv in rest_vars:
      var_dict[rv.op.name] = rv

    saver = tf.train.Saver(var_list=var_dict,
                           sharded=True,
                           max_to_keep=config.keep_checkpoint_max,
                           save_relative_paths=True)

  return saver

predict_saver = evaluate_saver
