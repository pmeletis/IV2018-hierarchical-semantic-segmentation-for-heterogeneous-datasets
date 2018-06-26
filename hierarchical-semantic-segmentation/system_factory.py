"""
Semantic Segmentation system.
"""

import copy
import functools
import glob
import collections
from datetime import datetime
from os.path import join, isdir, split, exists, isdir
from os import makedirs
import numpy as np
from PIL import Image
# from skimage.transform import resize as skimage_resize
from estimator.define_estimator import define_estimator
from utils.utils import _replacevoids, print_metrics_from_confusion_matrix

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.DEBUG)


class SemanticSegmentation(object):
  def __init__(self, input_fns, model_fn, settings=None, customconfig=None, params=None):
    # TODO: add input_fns, model_fn and settings checks
    assert customconfig is None and params is None, (
        'customconfig and params must remain None for now.')
    assert settings is not None, (
        'settings must be provided for now.')
    _validate_settings(settings)

    self._input_fns = input_fns
    self._model_fn = model_fn
    self._settings = copy.deepcopy(settings)
    self._estimator = None

    self._settings.training_lids2cids = _replacevoids(
        self._settings.training_problem_def['lids2cids'])
    if hasattr(self._settings, 'training_problem_def_citys'):
      self._settings.training_lids2cids_citys = _replacevoids(
          self._settings.training_problem_def_citys['lids2cids'])
    if hasattr(self._settings, 'training_problem_def_mapil'):
      self._settings.training_lids2cids_mapil = _replacevoids(
          self._settings.training_problem_def_mapil['lids2cids'])
    if hasattr(self._settings, 'training_problem_def_gtsdb'):
      self._settings.training_lids2cids_gtsdb = _replacevoids(
          self._settings.training_problem_def_gtsdb['lids2cids'])

    # construct candidate path for evaluation results directory in log directory,
    # with a unique counter index, e.g. if in log_dir/eval dir there exist
    # eval00, eval01, eval02, eval04 dirs it will create a new dir named eval05
    # TODO: better handle and warn for assumptions
    # for now it assums that only eval_ with 2 digits are present
    existing_eval_dirs = list(filter(isdir, glob.glob(join(self._settings.log_dir, 'eval_*'))))
    if existing_eval_dirs:
      existing_eval_dirs_names = [split(ed)[1] for ed in existing_eval_dirs]
      max_cnt = max([int(edn[-2:]) for edn in existing_eval_dirs_names])
    else:
      max_cnt = -1
    eval_res_dir = join(self._settings.log_dir, 'eval_' + f"{max_cnt + 1:02}")
    # save to settings for external access
    self._settings.eval_res_dir = eval_res_dir

  @property
  def settings(self):
    return self._settings

  def _create_estimator(self, runconfig):
    self._estimator = tf.estimator.Estimator(
        functools.partial(define_estimator, model_fn=self._model_fn),
        model_dir=self._settings.log_dir,
        config=runconfig,
        params=self._settings)

    return self._estimator

  def train(self):
    """Train the Semantic Segmentation model.
    """

    # create log dir
    if not tf.gfile.Exists(self._settings.log_dir):
      tf.gfile.MakeDirs(self._settings.log_dir)
      print('Created new logging directory:', self._settings.log_dir)

    # vars(args).items() returns (key,value) tuples from args.__dict__
    # and sorted uses first element of tuples to sort
    settings_dict = collections.OrderedDict(sorted(vars(self._settings).items()))

    # write configuration for future reference
    settings_filename = join(self._settings.log_dir, 'settings.txt')
    assert not exists(settings_filename), (f"Previous settings.txt found in "
        f"{self._settings.log_dir}. Rename it manually and restart training.")
    with open(settings_filename, 'w') as f:
      for k, v in enumerate(settings_dict):
        print(f"{k:2} : {v} : {settings_dict[v]}", file=f)

    # define the session_config
    if self._settings.enable_xla:
      session_config = tf.ConfigProto()
      # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
      # session_config = tf.ConfigProto(log_device_placement=True, gpu_options=gpu_options)
      # session_config.log_device_placement = True
      # session_config.gpu_options.per_process_gpu_memory_fraction = 0.95
      session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    else:
      session_config = None

    # Tensorflow internal error till at least r1.4
    # if keep_checkpoint_max is set to 0 or None doesn't do what it is supposed to do from docs
    runconfig = tf.estimator.RunConfig(
        model_dir=self._settings.log_dir,
        save_summary_steps=self._settings.save_summaries_steps,
        save_checkpoints_steps=self._settings.save_checkpoints_steps,
        session_config=session_config,
        keep_checkpoint_max=1000, # some big number to keeps all checkpoints
        log_step_count_steps=self._settings.save_summaries_steps)

    # create a local estimator
    self._create_estimator(runconfig)

    return self._estimator.train(
        input_fn=self._input_fns['train'],
        max_steps=self._settings.num_training_steps)

  def predict(self):
    if self._settings.Nb > 1:
      print('\nWARNING: during prediction only images with same shape (size and channels) '
            'are supported for batch size greater than one. In case of runtime error '
            'change batch size to 1.\n')
    if self._settings.enable_xla:
      session_config = tf.ConfigProto()
      session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    else:
      session_config = None

    runconfig = tf.estimator.RunConfig(
        model_dir=self._settings.log_dir,
        session_config=session_config)

    self._create_estimator(runconfig)

    predict_keys = copy.deepcopy(self._settings.predict_keys)
    # if void exists probabilities are needed (see _replace_void_labels)
    void_exists_lids2cids = -1 in self._settings.training_problem_def['lids2cids']
    if void_exists_lids2cids:
      predict_keys.append('probabilities')

    # maybe TF internal error, predict_keys should be deep copied internally
    predictions = self._estimator.predict(
        input_fn=self._input_fns['predict'],
        predict_keys=predict_keys,
        # if None latest checkpoint in self._settings.model_dir will be used
        checkpoint_path=self._settings.ckpt_path)

    # TODO: resize to system dimensions for the outputs

    # deal with void in training lids2cids
    if void_exists_lids2cids:
      if self._settings.replace_void_decisions:
        predictions = self._replace_void_labels(predictions)
      else:
        def _print_warning(predictions):
          for prediction in predictions:
            void_exists_decs = np.any(np.equal(prediction['decisions'],
                                               max(self._settings.training_lids2cids)))
            if void_exists_decs:
              print(f"\nWARNING: void class label ({max(self._settings.training_lids2cids)}) "
                    "exists in decisions.\n")
            yield prediction
        predictions = _print_warning(predictions)

      # delete 'probabilities' key since it was needed only locally for sanity
      def _delete_probs(predictions):
        for prediction in predictions:
          del prediction['probabilities']
          yield prediction
      predictions = _delete_probs(predictions)

    # deal with void in inference cids2lids
    if -1 in self._settings.inference_problem_def['cids2lids']:
      print('\nWARNING: -1 exists in cids2lids field of inference problem definition. '
            'For now it must me handled externally, and may cause outputs to have '
            '-1 labels.\n')

    # if predicting for different problem definition additional mapping is needed
    if self._settings.training_problem_def != self._settings.inference_problem_def:
      predictions = self._map_predictions_to_inference_problem_def(predictions)

    # resize to system dimensions: the output should have the provided system spatial size
    predictions = self._resize_decisions(predictions)

    return predictions

  def evaluate(self):

    eval_res_dir = self._settings.eval_res_dir
    print(f"\nWriting results in {eval_res_dir}.\n")
    makedirs(eval_res_dir)

    # write configuration for future reference
    # TODO: if settings.txt exists and train.py is re-run with different settings
    #   (e.g. logging settings) it is overwritten... (probably throw error)
    # if not tf.gfile.Exists(evalres_dir):
    #     tf.gfile.MakeDirs(evalres_dir)
    if exists(join(eval_res_dir, 'settings.txt')):
      print(f"WARNING: previous settings.txt in {eval_res_dir} is ovewritten.")
    with open(join(eval_res_dir, 'settings.txt'), 'w') as f:
      for k, v in vars(self._settings).items():
        print(f"{k} : {v}", file=f)

    if self._settings.enable_xla:
      session_config = tf.ConfigProto()
      session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    else:
      session_config = None

    runconfig = tf.estimator.RunConfig(
        model_dir=self._settings.log_dir,
        session_config=session_config)

    # create a local estimator
    self._create_estimator(runconfig)

    # get labels needed for online printing
    labels = self._settings.evaluation_problem_def['cids2labels']
    void_exists = -1 in self._settings.evaluation_problem_def['lids2cids']
    labels = labels[:-1] if void_exists else labels

    # evaluate given checkpoint or all checkpoints in log_dir
    # Note: eval_all_ckpts flag has priority over given checkpoint
    all_model_checkpoint_paths = [self._settings.ckpt_path]
    if self._settings.eval_all_ckpts:
      # sanity assignment
      self._settings.ckpt_path = None
      checkpoint_state = tf.train.get_checkpoint_state(self._settings.log_dir)
      # convert from protobuf repeatedScalarFieldContainer to list
      all_model_checkpoint_paths = list(checkpoint_state.all_model_checkpoint_paths)
      print(f"\n{len(all_model_checkpoint_paths)} checkpoint(s) will be evaluated.\n")

    all_metrics = []
    for cp in all_model_checkpoint_paths:
      # metrics contains only confusion matrix for now (and loss and global step)
      metrics = self._estimator.evaluate(
          input_fn=self._input_fns['eval'],
          steps=self._settings.num_eval_steps,
          # if None latest in model_dir will be used
          checkpoint_path=cp,
          name=split(eval_res_dir)[1][-2:])

      # deal with void in evaluation lids2cids
      if -1 in self._settings.evaluation_problem_def['lids2cids']:
        assert set(metrics.keys()) == {'global_step', 'loss', 'confusion_matrix'}, (
            'internal error: only confusion matrix metric is supported for mapping to '
            'a new problem definition for now. Change to training problem definition.')
        metrics['confusion_matrix'] = metrics['confusion_matrix'][:-1, :-1]

      # transform to different evaluation problem definition
      # if self._settings.training_problem_def != self._settings.evaluation_problem_def:
      #   metrics = self._map_metrics_to_evaluation_problem_def(metrics)

      #   # deal with void in training_cids2evaluation_cids
      #   if -1 in self._settings.evaluation_problem_def['training_cids2evaluation_cids']:
      #     assert set(metrics.keys()) == {'global_step', 'loss', 'confusion_matrix'}, (
      #         'internal error: only confusion matrix metric is supported for mapping to '
      #         'a new problem definition for now. Change to training problem definition.')
      #     metrics['confusion_matrix'] = metrics['confusion_matrix'][:-1, :-1]

      # online print the summary of metrics to terminal
      print_metrics_from_confusion_matrix(metrics['confusion_matrix'], labels, printcmd=True)

      all_metrics.append(metrics)

    return all_metrics

  def _replace_void_labels(self, predictions):
    # WARNING: time consuming function (due to argpartition), can take from 300 to 800 ms
    # enable it only when predicting for official evaluation/prediction

    # if void (-1) is provided in lids2cids, then the pixels that are predicted to belong
    # to the (internally) added void class should be labeled with the second most probable class
    # only decisions field is suppported for now
    accepted_keys = {'probabilities', 'decisions', 'rawimages', 'rawimagespaths'}
    predict_keys = self._settings.predict_keys
    # since 'probabilities' key is added temporarily it doesn't exist in _settings.predict_keys
    predict_keys.append('probabilities')
    for prediction in predictions:
      assert set(prediction.keys()).intersection(
          set(predict_keys)) == accepted_keys, (
              'internal error: only \'decisions\' predict_key is supported for mapping to '
              'a new problem definition for now. Change to training problem definition.')
      old_decs = prediction['decisions'] # 2D
      old_probs = prediction['probabilities'] # 3D
      void_decs_mask = np.equal(old_decs, max(self._settings.training_lids2cids))
      # implementing: values, indices = tf.nn.top_k(old_probs, k=2) # 3D
      # in numpy using argpartition for indices and
      # mesh grid and advanced indexing for values
      # argpartition returns np.int64
      top2_indices = np.argpartition(old_probs, -2)[..., -2] # 3D -> 2D
      # row, col = np.mgrid[0:old_probs.shape[0], 0:old_probs.shape[1]] # 2D, 2D
      # values = old_probs[row, col, indices]
      new_decs = np.where(void_decs_mask,
                          top2_indices.astype(np.int32, casting='same_kind'),
                          old_decs)
      prediction['decisions'] = new_decs
      yield prediction

  def _map_predictions_to_inference_problem_def(self, predictions):
    assert 'training_cids2inference_cids' in self._settings.inference_problem_def.keys(), (
        'Inference problem definition should have training_cids2inference_cids field, '
        'since provided inference problem definition file is not the same as training '
        'problem definition file.')

    tcids2pcids = np.array(_replacevoids(
        self._settings.inference_problem_def['training_cids2inference_cids']))

    for prediction in predictions:
      # only decisions is suppported for now
      assert set(prediction.keys()).intersection(
          set(self._settings.predict_keys)) == {'decisions', 'rawimages', 'rawimagespaths'}, (
              'internal error: only decisions predict_key is supported for mapping to '
              'a new problem definition for now. Change to training problem definition.')

      old_decisions = prediction['decisions']

      # TODO: add type and shape assertions
      assert old_decisions.ndim == 2, f"internal error: decisions shape is {old_decisions.shape}."

      new_decisions = tcids2pcids[old_decisions]
      if np.any(np.equal(new_decisions, -1)):
        print('WARNING: -1 label exists in decisions, handle it properly externally.')
        # raise NotImplementedError(
        #     'void mapping in different inference problem def is not yet implemented.')

      prediction['decisions'] = new_decisions

      yield prediction

  def _map_metrics_to_evaluation_problem_def(self, metrics):
    # if a net should be evaluated with problem that is not the problem with which it was
    # trained for, then the mappings from that problem should be provided.

    # only confusion_matrix is suppported for now
    assert set(metrics.keys()) == {'global_step', 'loss', 'confusion_matrix'}, (
        'internal error: only confusion matrix metric is supported for mapping to'
        'a new problem definition for now. Change to training problem definition.')
    assert 'training_cids2evaluation_cids' in self._settings.evaluation_problem_def.keys(), (
        'Evaluation problem definition should have training_cids2evaluation_cids field.')

    old_cm = metrics['confusion_matrix']
    tcids2ecids = np.array(_replacevoids(
        self._settings.evaluation_problem_def['training_cids2evaluation_cids']))

    # TODO: confusion matrix type and shape assertions
    assert old_cm.shape[0] == tcids2ecids.shape[0], f"Mapping lengths should be equal, {old_cm.shape}, {tcids2ecids.shape}."

    temp_shape = (max(tcids2ecids)+1, old_cm.shape[1])
    temp_cm = np.zeros(temp_shape, dtype=np.int64)

    # mas noiazei to kathe kainourio apo poio palio pairnei:
    #   i row of the new cm takes from rows of the old cm with indices:from_indices
    for i in range(temp_shape[0]):
      from_indices = [k for k, x in enumerate(tcids2ecids) if x == i]
      # print(from_indices)
      for fi in from_indices:
        temp_cm[i, :] += old_cm[fi, :].astype(np.int64)

    # oi grammes athroistikan kai tora tha athroistoun kai oi stiles
    new_shape = (max(tcids2ecids)+1, max(tcids2ecids)+1)
    new_cm = np.zeros(new_shape, dtype=np.int64)
    for j in range(new_shape[1]):
      from_indices = [k for k, x in enumerate(tcids2ecids) if x == j]
      # print(from_indices)
      for fi in from_indices:
        new_cm[:, j] += temp_cm[:, fi]

    metrics['confusion_matrix'] = new_cm

    return metrics

  def _resize_decisions(self, predictions):
    # resize decisions to system or input dimensions
    # only decisions is suppported for now
    # TODO: find a fast method without using PIL upsampling
    
    # new size defaults to provided values
    # if at least one is None then new size is the arbitrary size of rawimage in in step
    new_size = (self._settings.height_system, self._settings.width_system)
    is_arbitrary = not all(new_size)

    for prediction in predictions:
      assert set(prediction.keys()).intersection(
          set(self._settings.predict_keys)) == {'decisions', 'rawimages', 'rawimagespaths'}, (
              'internal error: only decisions predict_key is supported for mapping to '
              'a new problem definition for now. Change to training problem definition.')

      old_decisions = prediction['decisions']

      # TODO: add type and shape assertions
      assert old_decisions.ndim == 2, f"internal error: decisions shape is {old_decisions.shape}."

      if is_arbitrary:
        new_size = prediction['rawimages'].shape[:2]

      # save computation by comparing size
      if new_size != old_decisions.shape[:2]:
        new_decs = Image.fromarray(old_decisions).resize(reversed(new_size),
                                                         resample=Image.NEAREST)
        prediction['decisions'] = np.array(new_decs)

      yield prediction

def _validate_settings(settings):
  # TODO: add more validations

  assert settings.stride_system == 1 and settings.stride_network == 1, (
      'For now only stride of 1 is supported for stride_{system, network}.')

  assert all([settings.height_network == settings.height_feature_extractor,
              settings.width_network == settings.width_feature_extractor]), (
                  'For now {height, width}_{network, feature_extractor} should be equal.')

  # prediction specific
  if hasattr(settings, 'export_lids_images') and hasattr(settings, 'export_color_images'):
    if settings.export_lids_images or settings.export_color_images:
      assert settings.results_dir is not None and isdir(settings.results_dir), (
          'results_dir must a valid path if export_{lids, color}_images flags are True.')
