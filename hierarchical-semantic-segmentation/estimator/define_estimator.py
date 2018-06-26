import numpy as np
import copy
import os
import itertools
from operator import itemgetter
# tensorflow imports
import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tensorflow.python.client import timeline
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.training import create_train_op
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import resnet_utils
resnet_arg_scope = resnet_utils.resnet_arg_scope
# in TF 1.1 metrics_impl has _streaming_confusion_matrix hidden method
from tensorflow.python.ops import metrics_impl

from estimator.define_losses import define_losses
from estimator.define_savers import train_saver, predict_saver, evaluate_saver
from estimator.define_initializers import train_init
# from estimator.define_feature_extractor import define_feature_extractor
# from estimator.define_custom_metrics import confusion_matrices_for_classes_and_subclasses
from estimator.define_custom_metrics import mean_iou

from utils.utils import (almost_equal, get_unique_tensor_by_name_without_creating,
                         print_tensor_info, get_unique_variable_by_name_without_creating, get_saveable_objects_list, _replacevoids as replacevoids)


_ALLOWED_MODES = {tf.estimator.ModeKeys.TRAIN,
                  tf.estimator.ModeKeys.EVAL,
                  tf.estimator.ModeKeys.PREDICT}


def define_estimator(mode, features, labels, model_fn, config, params):
  """
  Assumptions:
    features: a dict containing rawfeatures and profeatures
      both: Nb x hf x wf x 3, tf.float32, in [0,1]
    labels: a dict containing rawfeatures and profeatures
      both: Nb x hf x wf, tf.int32, in [0,Nc-1]
  Args:
    features: First item returned by input_fn passed to train, evaluate, and predict.
    labels: Second item returned by input_fn passed to train, evaluate, and predict.
    mode: one of tf.estimator.ModeKeys.
  """

  assert mode in _ALLOWED_MODES, (
      'mode should be TRAIN, EVAL or PREDICT from tf.estimator.ModeKeys.')
  assert params.name_feature_extractor in {'resnet_v1_50', 'resnet_v1_101'}, (
      'params must have name_feature_extractor attribute in resnet_v1_{50,101}.')
  if params.name_feature_extractor == 'resnet_v1_101':
    raise NotImplementedError(
        'Use of resnet_v1_101 as base feature extractor is not yet implemented.')

  # unpack features
  rawimages = features['rawimages']
  proimages = features['proimages']
  # TODO: fix this temporary workaround for labels
  # rawlabels = labels['rawlabels'] if mode != tf.estimator.ModeKeys.PREDICT else None
  prolabels = labels['prolabels'] if mode != tf.estimator.ModeKeys.PREDICT else None

  print('debug:rawimages:', rawimages)
  print('debug:proimages:', proimages)
  print('debug:prolabels:', prolabels)

  ## build a fully convolutional model for semantic segmentation
  _, _, predictions = model_fn(mode, proimages, prolabels, config, params)

  # print('debug: predictions:', predictions)
  ## create training ops and exponential moving averages
  if mode == tf.estimator.ModeKeys.TRAIN:

    # global step
    global_step = tf.train.get_or_create_global_step()

    # losses
    with tf.variable_scope('losses'):
      losses = define_losses(mode, config, params, predictions, prolabels)

    # exponential moving averages
    # creates variables in checkpoint with name: 'emas/' + <variable_name> +
    #   {'ExponentialMovingAverage,Momentum}
    # ex.: for 'classifier/logits/Conv/biases' it saves also
    #          'emas/classifier/logits/Conv/biases/ExponentialMovingAverage'
    #      and 'emas/classifier/logits/Conv/biases/Momentum'
    # create_train_op guarantees to run GraphKeys.UPDATE_OPS collection
    #   before total_loss in every step, but doesn't give any guarantee
    #   for running after some other op, and since ema need to be run
    #   after applying the gradients maybe this code needs checking
    if params.ema_decay > 0:
      with tf.name_scope('exponential_moving_averages'):
        #for mv in slim.get_model_variables():
        #  print('slim.model_vars:', mv.op.name)
        ema = tf.train.ExponentialMovingAverage(params.ema_decay,
                                                num_updates=global_step,
                                                zero_debias=True)
        maintain_ema_op = ema.apply(var_list=tf.model_variables())
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, maintain_ema_op)

    # create training operation
    with tf.variable_scope('train_ops'):
      learning_rate = tf.train.piecewise_constant(global_step,
                                                  params.lr_boundaries,
                                                  params.lr_values)
      # optimizer
      if params.optimizer == 'SGDM':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            params.momentum,
            use_nesterov=params.use_nesterov)
      elif params.optimizer == 'SGD':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
      # training op
      train_op = create_train_op(
          losses['total'],
          optimizer,
          global_step=global_step,
          #update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS),
          # summarize_gradients=True,
          # #clip_gradient_norm=params.clip_grad_norm,
          # #gradient_multipliers=gradient_multipliers,
          check_numerics=False,
      )

    # TODO: maybe parameterize it
    training_hooks = [
      _RunMetadataHook(params.log_dir,
                       every_n_iter=max(params.num_training_steps//50,
                                        params.save_checkpoints_steps))]

    summaries_data = {'features': features,
                      'labels': labels,
                      'predictions': predictions,
                      'losses': losses,
                      'learning_rate': learning_rate}

  # flatten and concatenate decisions
  if mode in [tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT]:
    # don't forget to change confusion matrix outputs
    # C: 28, M: 66, G: 44, E: 71
    # flatten_decs = _flatten_all_decs(predictions['decisions'])
    # flatten_decs = _flatten_for_cityscapes_val(predictions['decisions'])
    # flatten_decs = _flatten_for_mapillary_val(predictions['decisions'])
    # flatten_decs = _flatten_for_cityscapes_extended_val(predictions['decisions'])
    flatten_decs = _flatten_for_gtsdb_val(predictions['decisions'])

  if mode == tf.estimator.ModeKeys.EVAL:
    with tf.variable_scope('losses'):
      losses = define_losses(mode, config, params, predictions, prolabels)

    # returns (variable, update_op)
    # TF internal error/problem: _streaming_confusion_matrix internally casts
    # labels and predictions to int64, and since we feed a dictionary, tensors are
    # passed by reference leading them to change type, thus we send an identity
    # confusion_matrix = metrics_impl._streaming_confusion_matrix(  # pylint: disable=protected-access
    #     tf.identity(prolabels),
    #     tf.identity(predictions['decisions']),
    #     params.training_Nclasses)
    confusion_matrix = metrics_impl._streaming_confusion_matrix(  # pylint: disable=protected-access
        prolabels,
        flatten_decs,
        44)

    # dict of metrics keyed by name with values tuples of (metric_tensor, update_op)
    # TODO: add more semantic segmentation metrics
    eval_metric_ops = {'confusion_matrix': (
        tf.to_int32(confusion_matrix[0]), confusion_matrix[1])}

  ## create EstimatorSpec according to mode
  # unpack predictions
  if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
    predictions = None
  else:
    # redefine predictions according to estimator requirements
    predictions = {
        'logits': predictions['logits'][0][0],
        'probabilities': predictions['probabilities'][0][0],
        # 'decisions': predictions['decisions'][0],
        'decisions': flatten_decs,
        }

  if mode == tf.estimator.ModeKeys.TRAIN:
    scaffold = _define_scaffold(mode, config, params, summaries_data)
    estimator_spec = tf.estimator.EstimatorSpec(
        mode,
        predictions=predictions,
        loss=losses['total'],
        train_op=train_op,
        training_hooks=training_hooks,
        scaffold=scaffold)
  elif mode == tf.estimator.ModeKeys.EVAL:
    scaffold = _define_scaffold(mode, config, params)
    estimator_spec = tf.estimator.EstimatorSpec(
        mode,
        predictions=predictions,
        loss=losses['total'],
        eval_metric_ops=eval_metric_ops,
        scaffold=scaffold)
  elif mode == tf.estimator.ModeKeys.PREDICT:
    scaffold = _define_scaffold(mode, config, params)
    # workaround for connecting input pipeline outputs to system output
    # TODO: make it more clear
    predictions['rawimages'] = rawimages
    predictions['rawimagespaths'] = features['rawimagespaths']
    # the expected predictions.keys() in this point is:
    # dict_keys(['logits', 'probabilities', 'decisions', 'rawimages', 'rawimagespaths'])
    estimator_spec = tf.estimator.EstimatorSpec(
        mode,
        predictions=predictions,
        scaffold=scaffold)

  return estimator_spec

def _define_scaffold(mode, config, params, summaries_data=None):
  """Creates scaffold containing initializers, savers and summaries.

  Args:
    summaries_data: dictionary containing all tensors needed for summaries during training

  Returns:
    a tf.train.Scaffold instance
  """
  # Comment: init_op with init_feed_dict, and init_fn are executed from SessionManager
  # only if model is not loaded successfully from checkpoint using the saver.
  # if no saver is provided then the default saver is constructed to load all
  # variables (from collections GLOBAL_VARIABLES and SAVEABLE_OBJECTS) and init_op won't
  # be executed.
  # For that reason, during training using init_checkpoint we provide a custom saver only
  # for model variables and an init_op to initialize all variables not in init_checkpoint.

  # create scopes outside of scaffold namescope
  with tf.name_scope('init') as init_scope:
    pass
  with tf.name_scope('saver') as saver_scope:
    pass

  with tf.name_scope('scaffold'):
    if mode == tf.estimator.ModeKeys.TRAIN:
      _define_summaries(mode, config, params, summaries_data)
      saver = train_saver(config, params, scope=saver_scope)
      init_op, init_feed_dict = train_init(config, params, scope=init_scope)
    elif mode == tf.estimator.ModeKeys.EVAL:
      saver = evaluate_saver(config, params, scope=saver_scope)
      init_op, init_feed_dict = [None]*2
    elif mode == tf.estimator.ModeKeys.PREDICT:
      saver = predict_saver(config, params, scope=saver_scope)
      init_op, init_feed_dict = [None]*2

    # WARNING: default ready_op and ready_for_local_init_op install operations
    #   in the graph to report_uninitialized_variables, resulting in too many ops,
    #   so make ready_for_local_init_op a no_op to reduce them.
    scaffold = tf.train.Scaffold(
        init_op=init_op,
        init_feed_dict=init_feed_dict,
        saver=saver)

  return scaffold

def _define_summaries(mode, config, params, summaries_data):
  del config
  assert mode == tf.estimator.ModeKeys.TRAIN, print('internal error: summaries only for training.')

  with tf.name_scope('summaries'), tf.device('/cpu:0'):
    # unpack necessary objects and tensors
    # WARNING: assumes all necessary items exist (maybe add assertions)
    rawimages = summaries_data['features']['rawimages']
    rawlabels = summaries_data['labels']['rawlabels']
    proimages = summaries_data['features']['proimages']
    prolabels = summaries_data['labels']['prolabels']
    _, probs, decs = itemgetter('logits', 'probabilities', 'decisions')(
        summaries_data['predictions'])
    tot_loss, reg_loss, seg_loss = itemgetter('total', 'regularization', 'segmentation')(
        summaries_data['losses'])
    # l1_seg_loss, l2_seg_loss_rider, l2_seg_loss_traffic_sign, l3_seg_loss_traffic_sign_front = seg_loss

    # drawing
    with tf.name_scope('drawing'):
      with tf.name_scope('palette'):
        palette = tf.constant(params.training_problem_def['cids2colors'], dtype=tf.uint8)
        palette_citys = tf.constant(params.training_problem_def_citys['cids2colors'], dtype=tf.uint8)
        palette_mapil = tf.constant(params.training_problem_def_mapil['cids2colors'], dtype=tf.uint8)
        palette_gtsdb = tf.constant(params.training_problem_def_gtsdb['cids2colors'], dtype=tf.uint8)

      # WARNING: assuming upsampling, that is all color_* images have the
      # same spatial dimensions
      # the common problem paletter coincides with mapillary palette
      flatten_decs = _flatten_all_decs(decs)
      color_decisions = _cids2col(flatten_decs, palette)
      # generate confidence image, preventing TF from normalizing max prob
      # to 1, by casting to tf.uint8
      # color_confidences = tf.stack([tf.cast(tf.reduce_max(probs, axis=3)*255, tf.uint8)]*3, axis=3)
      color_prolabels = tf.concat(
          [_cids2col(prolabels[0:1], palette_citys),
           _cids2col(prolabels[1:3], palette_mapil),
           _cids2col(prolabels[3:4], palette_gtsdb)],
          0)

      # tf.summary.image('rawimages', tf.image.convert_image_dtype(rawimages, tf.uint8), max_outputs=3, family='raw_data')
      # tf.summary.image('rawlabels', _cids2col(rawlabels, palette), max_outputs=3, family='raw_data')
      tf.summary.image('proimages', tf.image.convert_image_dtype(proimages, tf.uint8, saturate=True), max_outputs=4, family='preprocessed_data')
      tf.summary.image('prolabels', color_prolabels, max_outputs=4, family='preprocessed_data')
      tf.summary.image('decisions', color_decisions, max_outputs=4, family='results')
      # tf.summary.image('confidences', color_confidences, max_outputs=params.Nb_mapil, family='results')

      # compute batch metrics
      # m_iou = mean_iou(prolabels, decs[0], num_classes=7+1, params=params)

    # TODO: in order to disable loss summary created internally by estimator this line should
    # evaluate to False:
    # not any([x.op.name == 'loss' for x in ops.get_collection(ops.GraphKeys.SUMMARIES)])
    tf.summary.scalar('total', tot_loss, family='losses')
    tf.summary.scalar('regularization', reg_loss, family='losses')
    tf.summary.scalar('l1_segmentation', seg_loss[0][0], family='losses')
    tf.summary.scalar(f"l2_segmentation_driveable", seg_loss[1][0], family='losses')
    tf.summary.scalar(f"l2_segmentation_rider", seg_loss[1][1], family='losses')
    tf.summary.scalar(f"l2_segmentation_traffic_sign", seg_loss[1][2], family='losses')
    tf.summary.scalar(f"l3_segmentation_traffic_sign_front", seg_loss[2][0], family='losses')
    # tf.summary.scalar('mean_IOU', m_iou, family='metrics')

    tf.summary.scalar('learning_rate', summaries_data['learning_rate'], family='optimizer')

def _cids2col(cids, palette):
  # cids: Nb x H x W, tf.int32, with class ids in [0,Nc-1]
  # palette: Nc x 3, tf.uint8, with rgb colors in [0,255]
  # returns: Nb x H x W x 3, tf.uint8, in [0,255]

  # TODO: add type checking
  return tf.gather_nd(palette, cids[..., tf.newaxis])

class _RunMetadataHook(tf.train.SessionRunHook):
  """Exports the run metadata as a trace to log_dir every N local steps or every N seconds.
  """
  # TODO: implement this with tf.profiler

  def __init__(self, log_dir, every_n_iter=None, every_n_secs=None):
    """Initializes a `_RunMetadataHook`.

    Args:
      log_dir: the log_dir directory to save traces.
      every_n_iter: `int`, save traces once every N local steps.
      every_n_secs: `int` or `float`, save traces once every N seconds.

      Exactly one of `every_n_iter` and `every_n_secs` should be provided.

    Raises:
      ValueError: if `every_n_iter` is non-positive.
    """
    if (every_n_iter is None) == (every_n_secs is None):
      raise ValueError("Exactly one of every_n_iter and every_n_secs must be provided.")
    if every_n_iter is not None and every_n_iter <= 0:
      raise ValueError(f"Invalid every_n_iter={every_n_iter}.")
    self._timer = tf.train.SecondOrStepTimer(every_secs=every_n_secs, every_steps=every_n_iter)
    self._iter_count = None
    self._should_trigger = None
    self._tf_global_step = None
    self._np_global_step = None
    self._log_dir = log_dir

  def begin(self):
    self._timer.reset()
    self._iter_count = 0

  def after_create_session(self, session, coord):  # pylint: disable=unused-argument
    self._tf_global_step = tf.train.get_global_step()
    assert self._tf_global_step, 'Internal error: RunMetadataHook cannot retrieve global step.'

  def before_run(self, run_context):  # pylint: disable=unused-argument
    self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)
    if self._should_trigger:
      self._timer.update_last_triggered_step(self._iter_count)
      return tf.train.SessionRunArgs(
          fetches=self._tf_global_step,
          options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE))
    else:
      return None

  def after_run(self, run_context, run_values):  # pylint: disable=unused-argument
    if self._should_trigger:
      self._np_global_step = run_values.results
      # self._iter_count = self._np_global_step
      self._timer.update_last_triggered_step(self._iter_count)
      run_metadata = run_values.run_metadata
      if run_metadata is not None:
        tl = timeline.Timeline(run_metadata.step_stats)
        trace = tl.generate_chrome_trace_format()
        trace_filename = os.path.join(self._log_dir, f"tf_trace-{self._np_global_step}.json")
        tf.logging.info(f"Writing trace to {trace_filename}.")
        file_io.write_string_to_file(trace_filename, trace)
        # TODO: add run_metadata to summaries with summary_writer
        #   find how summaries are saved in the estimator and add them
        # summary_writer.add_run_metadata(run_metadata, f"run_metadata-{self._global_step}")

    self._iter_count += 1

def _have_compatible_shapes(lot):
  # lot: list_of_tensors
  tv = True
  for t1, t2 in itertools.combinations(lot, 2):
    tv = tv and t1.shape.is_compatible_with(t2.shape)
  return tv

def _have_equal_shapes(lot):
  # lot: list_of_tensors
  tv = True
  for t1, t2 in itertools.combinations(lot, 2):
    tv = tv and (t1.shape == t2.shape)
  return tv

def _flatten_all_decs(decisions):
  # this implements:
  # 1) the hierarchical decision rule for concatenating all levels of decisions
  # 2) it doesn't implement any mapping for decisions to different classes

  # TODO: implement second most probable class for unlabeled pixels
  # the super label space is the mapillary label space (since cityscapes labels are included)
  # l1: 53+1 classes
  # l2: [10+1, 3+1, 2+1] classes
  l1_decs, l2_decs, l3_decs = decisions
  l2_decs_driveable, l2_decs_rider, l2_decs_traffic_sign = l2_decs

  # unlabeled_tensor = tf.ones_like(l1_decs)*108
  # l1_Nclasses = 53+1
  # l2_Nclasses = [10+1, 3+1, 2+1]
  # l3_Nclasses = [43+1]

  l1_cids2common_cids = [0, 1, 2, 3, 4, 5, 6, 43, 9, 11, 12, 15, 16, 17, 18, 19, 22, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 42, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 108]
  l2_cids_driveable2common_cids = [7, 8, 10, 13, 14, 23, 24, 36, 41, 43, 108]
  l2_cids_rider2common_cids = [20, 21, 22, 108]
  l2_cids_traffic_sign2common_cids = [49, 50, 108]
  l3_cids2common_ids = [65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108]

  # 1) fix offsets to common ids and unlabeled id
  # adding offsets
  l1_decs2common_decs = tf.gather(tf.to_int32(l1_cids2common_cids), l1_decs[0])
  l2_decs_driveable2common_decs = tf.gather(tf.to_int32(l2_cids_driveable2common_cids), l2_decs_driveable)
  l2_decs_rider2common_decs = tf.gather(tf.to_int32(l2_cids_rider2common_cids), l2_decs_rider)
  l2_decs_traffic_sign2common_decs = tf.gather(tf.to_int32(l2_cids_traffic_sign2common_cids), l2_decs_traffic_sign)
  l3_decs2common_decs = tf.gather(tf.to_int32(l3_cids2common_ids), l3_decs[0])

  # 2) flatten hierarchy
  # the condition is on per classifier cids, because the common cids are ambiguous in l1 and l2 levels
  # 7: driveable, 16: rider, 38: traffic sign
  flatten_decs = \
      tf.where(tf.equal(l1_decs[0], 7),
               l2_decs_driveable2common_decs,
               tf.where(tf.equal(l1_decs[0], 16),
                        l2_decs_rider2common_decs,
                        tf.where(tf.equal(l1_decs[0], 38),
                                 tf.where(tf.equal(l2_decs_traffic_sign, 1),
                                          l3_decs2common_decs,
                                          l2_decs_traffic_sign2common_decs),
                                 l1_decs2common_decs)))

  return flatten_decs

def _flatten_for_cityscapes_val(decisions):
  # flat and map for cityscapes evaluation to 27+1 classes (not on the official benchmark classes)
  # this implements HDR (hier. dec. rule) using L1 classes and L2 driveable and traffic sign classes (since cityscapes names as traffic signs only the front of traffic signs)
  # maps combined hierarchy class ids to cityscapes class ids

  l1_decs, l2_decs, _ = decisions
  l2_decs_driveable, _, l2_decs_traffic_sign = l2_decs

  l1_cids2cityscapes_cids = [27, 27, 27, 7, 8, 27, 6, 1, 27, 27, 4, 2, 9, 5, 10, 17, 18, 27, 27, 16, 27, 15, 14, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 11, 27, 27, 12, 13, 27, 26, 27, 21, 19, 22, 25, 24, 27, 23, 20, 27, 27, 0, 27]
  l2_cids_driveable2cityscapes_cids = [1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 27]
  l2_cids_traffic_sign2cityscapes_cids = [27, 13, 27]

  # 1) add offsets to common cids
  l1_decs2cityscapes_decs = tf.gather(tf.to_int32(l1_cids2cityscapes_cids), l1_decs[0])
  l2_decs_driveable2cityscapes_decs = tf.gather(tf.to_int32(l2_cids_driveable2cityscapes_cids), l2_decs_driveable)
  l2_decs_traffic_sign2cityscapes_decs = tf.gather(tf.to_int32(l2_cids_traffic_sign2cityscapes_cids), l2_decs_traffic_sign)

  # 2) flatten hierarchy
  flatten_decs = tf.where(tf.equal(l1_decs[0], 7),
                          l2_decs_driveable2cityscapes_decs,
                          tf.where(tf.equal(l1_decs[0], 38),
                                   l2_decs_traffic_sign2cityscapes_decs,
                                   l1_decs2cityscapes_decs))

  return flatten_decs

def _flatten_for_mapillary_val(decisions):
  # flat and map for mapillary vistas evaluation to 65+1 classes
  # this implements HDR (hier. dec. rule) using L1 classes and L2 classes (since mapillary has class till L2 of the hierarchy)
  # maps combined hierarchy class ids to mapillary class ids

  l1_decs, l2_decs, _ = decisions
  l2_decs_driveable, l2_decs_rider, l2_decs_traffic_sign = l2_decs

  l1_cids2mapillary_cids = [0, 1, 2, 3, 4, 5, 6, 43, 9, 11, 12, 15, 16, 17, 18, 19, 22, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 42, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]
  l2_cids_driveable2mapillary_cids = [7, 8, 10, 13, 14, 23, 24, 36, 41, 43, 65]
  l2_cids_rider2mapillary_cids = [20, 21, 22, 65]
  l2_cids_traffic_sign2mapillary_cids = [49, 50, 65]

  # 1) add offsets to common cids
  l1_decs2mapillary_decs = tf.gather(tf.to_int32(l1_cids2mapillary_cids), l1_decs[0])
  l2_decs_driveable2mapillary_decs = tf.gather(tf.to_int32(l2_cids_driveable2mapillary_cids), l2_decs_driveable)
  l2_decs_rider2mapillary_decs = tf.gather(tf.to_int32(l2_cids_rider2mapillary_cids), l2_decs_rider)
  l2_decs_traffic_sign2mapillary_decs = tf.gather(tf.to_int32(l2_cids_traffic_sign2mapillary_cids), l2_decs_traffic_sign)

  # 2) flatten hierarchy
  # rided l1 id: 20, traffic sign l1 id: 47
  flatten_decs = \
      tf.where(tf.equal(l1_decs[0], 7),
               l2_decs_driveable2mapillary_decs,
               tf.where(tf.equal(l1_decs[0], 16),
                        l2_decs_rider2mapillary_decs,
                        tf.where(tf.equal(l1_decs[0], 38),
                                 l2_decs_traffic_sign2mapillary_decs,
                                 l1_decs2mapillary_decs)))

  return flatten_decs

def _flatten_for_cityscapes_extended_val(decisions):
  # flat and map for cityscapes extended evaluation to 70+1 classes (not on the official benchmark classes)
  # this implements HDR (hier. dec. rule) using L1 classes, L2 traffic sign classes and L3 traffic sign front classes
  # maps combined hierarchy class ids to cityscapes extended class ids

  l1_decs, l2_decs, l3_decs = decisions
  l2_decs_driveable, l2_decs_rider, l2_decs_traffic_sign = l2_decs

  l1_cids2cityscapes_extended_cids = [70, 70, 70, 7, 8, 70, 6, 1, 70, 70, 4, 2, 9, 5, 10, 17, 18, 70, 70, 16, 70, 15, 14, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 11, 70, 70, 12, 13, 70, 26, 70, 21, 19, 22, 25, 24, 70, 23, 20, 70, 70, 0, 70]
  l2_cids_driveable2cityscapes_extended_cids = [1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 70]
  l2_cids_traffic_sign2cityscapes_extended_cids = [70, 13, 70]
  l3_cids2cityscapes_extended_cids = [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]

  # 1) add offsets to common cids
  l1_decs2cityscapes_extended_decs = tf.gather(tf.to_int32(l1_cids2cityscapes_extended_cids), l1_decs[0])
  l2_decs_driveable2cityscapes_extended_decs = tf.gather(tf.to_int32(l2_cids_driveable2cityscapes_extended_cids), l2_decs_driveable)
  l2_decs_traffic_sign2cityscapes_extended_decs = tf.gather(tf.to_int32(l2_cids_traffic_sign2cityscapes_extended_cids), l2_decs_traffic_sign)
  l3_decs2cityscapes_extended_decs = tf.gather(tf.to_int32(l3_cids2cityscapes_extended_cids), l3_decs[0])

  # 2) flatten hierarchy
  flatten_decs = \
      tf.where(tf.equal(l1_decs[0], 7),
               l2_decs_driveable2cityscapes_extended_decs,
               tf.where(tf.equal(l1_decs[0], 38),
                        l3_decs2cityscapes_extended_decs,
                        l1_decs2cityscapes_extended_decs))

  return flatten_decs

def _flatten_for_gtsdb_val(decisions):
  # flat and map for GTSDB evaluation to 43+1 classes
  # this implements HDR (hier. dec. rule) using L1 classes, L2 traffic sign classes and L3 traffic sign front classes
  # maps combined hierarchy class ids to cityscapes extended class ids

  l1_decs, l2_decs, l3_decs = decisions
  _, _, l2_decs_traffic_sign = l2_decs

  # l1_cids2gtsdb_cids = [43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43]
  # l2_cids_traffic_sign2gtsdb_cids = [0, 0, 43]
  # l3_cids2gtsdb_cids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]

  # 1) add offsets to common cids
  # l1_decs2gtsdb_decs = tf.gather(tf.to_int32(l1_cids2gtsdb_cids), l1_decs[0])
  # l3_decs2gtsdb_decs = tf.gather(tf.to_int32(l3_cids2gtsdb_cids), l3_decs[0])

  # 2) flatten hierarchy
  flatten_decs = \
      tf.where(tf.equal(l1_decs[0], 38),
               l3_decs[0],
               tf.ones_like(l3_decs[0]) * 43)

  return flatten_decs
