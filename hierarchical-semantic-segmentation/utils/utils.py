
import copy, argparse, collections, json, os, glob, sys

from scipy import misc
import moviepy.editor as mp
import numpy as np

import tensorflow as tf


class SemanticSegmentationArguments(object):
  """Example class for how to collect arguments for command line execution.
  """
  # _DEFAULT_PROBLEM = 'problem01'

  def __init__(self):
    self._parser = argparse.ArgumentParser()
    self.add_general_arguments()
    self.add_tf_arguments()

  def parse_args(self, argv):
    # parse all arguments and add manually additional arguments
    self.args = self._parser.parse_args(argv)

    # problem definitions
    self.args.training_problem_def = json.load(
        open(self.args.training_problem_def_path, 'r'))
    if hasattr(self.args, 'training_problem_def_path_citys'):
      self.args.training_problem_def_citys = json.load(
          open(self.args.training_problem_def_path_citys, 'r'))
    if hasattr(self.args, 'training_problem_def_path_mapil'):
      self.args.training_problem_def_mapil = json.load(
          open(self.args.training_problem_def_path_mapil, 'r'))
    if hasattr(self.args, 'training_problem_def_path_gtsdb'):
      self.args.training_problem_def_gtsdb = json.load(
          open(self.args.training_problem_def_path_gtsdb, 'r'))
    if hasattr(self.args, 'evaluation_problem_def_path'):
      if self.args.evaluation_problem_def_path is None:
        assert False
        # self.args.evaluation_problem_def = self.args.training_problem_def
      else:
        self.args.evaluation_problem_def = json.load(
            open(self.args.evaluation_problem_def_path, 'r'))
    if hasattr(self.args, 'inference_problem_def_path'):
      if self.args.inference_problem_def_path is None:
        # TODO: implement the use of inference problem_def for training results in all functions
        self.args.inference_problem_def = self.args.training_problem_def
      else:
        self.args.inference_problem_def = json.load(
            open(self.args.inference_problem_def_path, 'r'))

    return self.args

  def add_general_arguments(self):
    # hs -> ||                                 SYSTEM                                       || -> hs/s
    # hs -> || hn -> ||                   LEARNABLE NETWORK                     || -> hn/sn || -> hs/s
    # hs -> || hn -> ||  hf -> FEATURE EXTRACTOR -> hf/sfe -> [UPSAMPLER -> hf] || -> hn/sn || -> hs/s
    # input || image || [tile] -> batch ->               supervision -> [stich] || labels    || output
    self._parser.add_argument('--stride_system', type=int, default=1, help='Output stride of the system. Use 1 for same input and output dimensions.')
    self._parser.add_argument('--stride_network', type=int, default=1, help='Output stride of the network. Use in case labels have different dimensions than output of learnable network.')
    self._parser.add_argument('--stride_feature_extractor', type=int, default=8, help='Output stride of the feature extractor. For the resnet_v1_* familly must be in {4,8,16,...}.')
    self._parser.add_argument('--name_feature_extractor', type=str, default='resnet_v1_50', choices=['resnet_v1_50', 'resnet_v1_101'], help='Feature extractor network.')
    self._parser.add_argument('--height_system', type=int, default=None,
                              help='Height of input images to the system. If None arbitrary height is supported.')
    self._parser.add_argument('--width_system', type=int, default=None,
                              help='Width of input images to the system. If None arbitrary width is supported.')
    self._parser.add_argument('--height_network', type=int, default=512, help='Height of input images to the trainable network.')
    self._parser.add_argument('--width_network', type=int, default=706, help='Width of input images to the trainable network.')
    self._parser.add_argument('--height_feature_extractor', type=int, default=512, help='Height of feature extractor images. If height_feature_extractor != height_network then it must be its divisor for patch-wise training.')
    self._parser.add_argument('--width_feature_extractor', type=int, default=706, help='Width of feature extractor images. If width_feature_extractor != width_network then it must be its divisor for patch-wise training.')
    
    # 1024 -> ||                                   ALGORITHM                                 || -> 1024  ::  h=512, s=512/512=1
    # 1024 -> || 1024 -> ||                     LEARNABLE NETWORK                 || -> 1024 || -> 1024  ::  hl=512, snet=512/512=1
    # 1024 -> || 1024 -> || 512 -> FEATURE EXTRACTOR -> 128 -> [UPSAMPLER -> 512] || -> 1024 || -> 1024  ::  hf=512, sfe=512/128=4
    self._parser.add_argument('--feature_dims_decreased', type=int, default=256, help='If >0 decreases feature dimensions of the feature extractor\'s output (usually 2048) to feature_dims_decreased using another convolutional layer.')
    self._parser.add_argument('--fov_expansion_kernel_size', type=int, default=0, help='If >0 increases the Field of View of the feature representation using an extra convolutional layer with this kernel.')
    self._parser.add_argument('--fov_expansion_kernel_rate', type=int, default=0, help='If >0 increases the Field of View of the feature representation using an extra convolutional layer with this dilation rate.')
    self._parser.add_argument('--upsampling_method', type=str, default='hybrid', choices=['no', 'bilinear', 'hybrid'], help='No, Bilinear or hybrid upsampling are currently supported.')
    # self._parser.add_argument('--subnet_experimenting', action='store_true', help='Temporary flag for subnet experimenting.')

  def add_tf_arguments(self):
    # general flags
    self._parser.add_argument('--enable_xla', action='store_true', help='Whether to enable XLA accelaration.')

  def add_train_arguments(self):
    """Arguments for training.

    TFRecords requirements...
    """
    # general configuration flags (in future will be saved in otherconfig)
    self._parser.add_argument('log_dir', type=str, default='...',
                              help='Directory for saving checkpoints, settings, graphs and training statistics.')
    # self._parser.add_argument('tfrecords_path_citys', type=str, 
    #                           default='/media/panos/data/datasets/cityscapes/tfrecords/trainFine.tfrecords',
    #                           help='Training is supported only from TFRecords. Refer to help for the mandatory fields for examples inside tfrecords.')
    self._parser.add_argument('--tfrecords_path_citys', type=str, 
                              default='/media/panos/data/datasets/cityscapes/tfrecords/trainFine_v4.tfrecords',
                              help='Training is supported only from TFRecords. Refer to help for the mandatory fields for examples inside tfrecords.')
    self._parser.add_argument('--tfrecords_path_mapil', type=str, 
                              default='/media/panos/data/datasets/mapillary/mapillary-vistas-dataset_public_v1.0/tfrecords/train.tfrecord',
                              help='Training is supported only from TFRecords. Refer to help for the mandatory fields for examples inside tfrecords.')
    self._parser.add_argument('--tfrecords_path_gtsdb', type=str, 
                              default='/media/panos/data/datasets/gtsdb/tfrecords/trainCoarse.tfrecords',
                              help='Training is supported only from TFRecords. Refer to help for the mandatory fields for examples inside tfrecords.')
    self._parser.add_argument('--Ntrain_citys', type=int, default=2975,
                              help='Temporary parameter for the number of training examples.')
    self._parser.add_argument('--Ntrain_mapil', type=int, default=18000,
                              help='Temporary parameter for the number of training examples.')
    self._parser.add_argument('--Ntrain_gtsdb', type=int, default=600,
                              help='Temporary parameter for the number of training examples.')
    self._parser.add_argument('--init_ckpt_path', type=str, default='/home/panos/tensorflow/panos/panos/ResNet_v1/tensorflow_models/resnet_v1_50_official.ckpt',
                              help='If provided and log_dir is empty, same variables between checkpoint and the model will be initiallized from this checkpoint. Otherwise, training will continue from the latest checkpoint in log_dir according to tf.Estimator. If you want to initialize partially from this checkpoint delete of modify names of variables in the checkpoint.')
    self._parser.add_argument('--training_problem_def_path', type=str,
                              default= '/home/panos/git/hierarchical-semantic-segmentation-2/semantic-segmentation/training_problem_def.json',
                              help='Problem definition json file. For required fields refer to help.')
    self._parser.add_argument('--training_problem_def_path_citys', type=str,
                              default= '/media/panos/data/datasets/cityscapes/panos/jsons/' + 'problem03' + '.json',
                              help='Problem definition json file. For required fields refer to help.')
    self._parser.add_argument('--training_problem_def_path_mapil', type=str,
                              default= '/media/panos/data/datasets/mapillary/mapillary-vistas-dataset_public_v1.0/panos/jsons/' + 'problem01' + '.json',
                              help='Problem definition json file. For required fields refer to help.')
    self._parser.add_argument('--training_problem_def_path_gtsdb', type=str,
                              default= '/media/panos/data/datasets/gtsdb/panos/jsons/problem01.json',
                              help='Problem definition json file. For required fields refer to help.')
    # self._parser.add_argument('--inference_problem_def_path', type=str, default=None,
    #                           help='Problem definition json file for inference. If provided it will be used instead of training_problem_def for inference. For required fields refer to help.')
    self._parser.add_argument('--save_checkpoints_steps', type=int, default=2000,
                              help='Save checkpoint every save_checkpoints_steps steps.')
    self._parser.add_argument('--save_summaries_steps', type=int, default=120,
                              help='Save summaries every save_summaries_steps steps.')
    # self._parser.add_argument('--collage_image_summaries', action='store_true', help='Whether to collage input and result images in summaries. Inputs and results won\'t be collaged if they have different sizes.')

    # optimization and losses flags (in future will be saved in hparams)
    self._parser.add_argument('--Ne', type=int, default=17, help='Number of epochs to train for, according to biggest dataset.')
    self._parser.add_argument('--Nb_citys', type=int, default=1, help='Number of examples per batch.')
    self._parser.add_argument('--Nb_mapil', type=int, default=2, help='Number of examples per batch.')
    self._parser.add_argument('--Nb_gtsdb', type=int, default=1, help='Number of examples per batch.')
    self._parser.add_argument('--learning_rate_schedule', type=str, default='piecewise_constant', choices=['piecewise_constant'], help='Learning rate schedule.')
    self._parser.add_argument('--learning_rate_initial', type=float, default=0.01, help='Initial learning rate.')
    self._parser.add_argument('--learning_rate_decay', type=float, default=0.5, help='Decay rate for learning rate.')
    self._parser.add_argument('--optimizer', type=str, default='SGDM', choices=['SGD', 'SGDM'], help='Stochastic Gradient Descent optimizer with or without Momentum.')
    self._parser.add_argument('--batch_norm_decay', type=float, default=0.9, help='BatchNorm decay (decrease when using smaller batch (Nb or image dims)).')
    self._parser.add_argument('--ema_decay', type=float, default=0.9, help='If >0 additionally save exponential moving averages of training variables with this decay rate.')
    self._parser.add_argument('--regularization_weight', type=float, default=0.00017, help='Weight for the L2 regularization losses in the total loss (decrease when using smaller batch (Nb or image dims)).')
    # only for Momentum optimizer
    self._parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGDM.')
    self._parser.add_argument('--use_nesterov', action='store_true', help='Enable Nesterov acceleration for SGDM.')

  def add_predict_arguments(self):

    # saved model arguments: log_dir, [ckppt_path], training_problem_def_path
    self._parser.add_argument('log_dir', type=str, default=None,
                              help='Logging directory containing the trained model checkpoints and settings. The latest checkpoint will be loaded from this directory by default, unless ckpt_path is provided.')
    self._parser.add_argument('--ckpt_path', type=str, default=None,
                              help='If provided, this checkpoint (if exists) will be used.')
    self._parser.add_argument('--training_problem_def_path', type=str,
                              default='training_problem_def.json',
                              help='Problem definition json file. For required fields refer to help.')
    
    # inference arguments: prediction_dir, [results_dir], [inference_problem_def_path],
    #                      [plotting], [export_color_images], [export_lids_images]
    self._parser.add_argument('predict_dir', type=str, default=None,
                              help='Directory to scan for media recursively and to write results under created results directory with the same directory structure. For supported media files check help.')
    self._parser.add_argument('--results_dir', type=str, default=None,
                              help='If provided results will be written to this directory.')
    self._parser.add_argument('--inference_problem_def_path', type=str, default=None,
                              help='Problem definition json file for inference. If provided it will be used instead of training_problem_def. For required fields refer to help.')
    self._parser.add_argument('--plotting', action='store_true',
                              help='Whether to plot results.')
    self._parser.add_argument('--timeout', type=float, default=10.0,
                              help='Timeout for continuous plotting, if plotting flag is provided.')
    self._parser.add_argument('--export_color_images', action='store_true',
                              help='Whether to export color image results.')
    self._parser.add_argument('--export_lids_images', action='store_true',
                              help='Whether to export label ids image results. Label ids are defined in {training,inference}_problem_def_path.')
    self._parser.add_argument('--replace_void_decisions', action='store_true',
                              help='Whether to replace void labeled pixels with the second most probable class (effective only when void (-1) is provided in lids2cids field in training problem definition). Enable only for official prediction/evaluation as it uses a time consuming function.')

    # SemanticSegmentation and system arguments:  [Nb], [restore_emas]
    self._parser.add_argument('--Nb', type=int, default=1,
                              help='Number of examples per batch.')
    self._parser.add_argument('--restore_emas', action='store_true',
                              help='Whether to restore exponential moving averages instead of normal last step\'s saved variables.')

    # consider for adding in the future arguments
    # self._parser.add_argument('--export_probs', action='store_true', help='Whether to export probabilities results.')
    # self._parser.add_argument('--export_for_algorithm_evaluation', action='store_true', help='Whether to plot and export using the algorithm input size (h,w).')

  def add_evaluate_arguments(self):
    self._parser.add_argument('log_dir', type=str, default=None,
                              help='Logging directory containing the trained model checkpoints and settings. The latest checkpoint will be evaluated from this directory by default, unless ckpt_path or evall_all_ckpts are provided.')
    self._parser.add_argument('--eval_all_ckpts', action='store_true',
                              help='Whether to evaluate all checkpoints in log_dir. It has priority over --ckpt_path argument.')
    self._parser.add_argument('--ckpt_path', type=str, default=None,
                              help='If provided, this checkpoint (if exists) will be evaluated.')
    self._parser.add_argument('tfrecords_path', type=str, default='/media/panos/data/datasets/cityscapes/tfrecords/valFine.tfrecords',
                              help='Evaluation is supported only from TFRecords. Refer to help for the mandatory fields for examples inside tfrecords.')
    self._parser.add_argument('Neval', type=int, default=500,
                              help='Temporary parameter for the number of evaluated examples.')
    self._parser.add_argument('--training_problem_def_path', type=str,
                              default= '/home/panos/git/hierarchical-semantic-segmentation-2/semantic-segmentation/training_problem_def.json',
                              help='Problem definition json file. For required fields refer to help.')
    self._parser.add_argument('--training_problem_def_path_citys', type=str,
                              default= '/media/panos/data/datasets/cityscapes/panos/jsons/' + 'problem03' + '.json',
                              help='Problem definition json file. For required fields refer to help.')
    self._parser.add_argument('--training_problem_def_path_mapil', type=str,
                              default= '/media/panos/data/datasets/mapillary/mapillary-vistas-dataset_public_v1.0/panos/jsons/' + 'problem01' + '.json',
                              help='Problem definition json file. For required fields refer to help.')
    self._parser.add_argument('evaluation_problem_def_path', type=str, default=None,
                              help='Problem definition json file for evaluation. If provided it will be used instead of training_problem_def. For required fields refer to help.')

    # self._parser.add_argument('eval_steps', type=int, help='500 for cityscapes val, 2975 for cityscapes train.')
    self._parser.add_argument('--Nb', type=int, default=1,
                              help='Number of examples per batch.')
    self._parser.add_argument('--restore_emas', action='store_true',
                              help='Whether to restore exponential moving averages instead of normal last step\'s saved variables.')
    
    # consider for future
    # self._parser.add_argument('--results_dir', type=str, default=None,
    #                           help='If provided evaluation results will be written to this directory, otherwise they will be written under a created results directory under log_dir.')

def print_tensor_info(tensor):
  print(f"debug:{tensor.op.name}: {tensor.get_shape().as_list()} {tensor.dtype}")

def get_unique_variable_by_name_without_creating(name):
  variables = [v for v in tf.global_variables() + tf.local_variables() if name==v.op.name]
  assert len(variables)==1, f"Found {len(variables)} variables for name {name}."
  return variables[0]

def get_unique_tensor_by_name_without_creating(name):
  tensors = [t for t in tf.contrib.graph_editor.get_tensors(tf.get_default_graph()) if name==t.name]
  assert len(tensors)==1, f"Found {len(tensors)} tensors."
  return tensors[0]

def get_saveable_objects_list(graph):
  return (graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) + graph.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS))

def ids2image(ids, palette):
  # ids: Nb x H x W, with elements in [0,K-1]
  # palette: K x 3, tf.uint8
  # returns: Nb x H x W x 3, tf.uint8
  
  # TODO: add type checking
  return tf.gather_nd(palette, tf.expand_dims(ids, axis=-1))

# def convert_ids(label, mode, params):
#   # label: TF tensor: H x W, tf.int32, [0,tf.int32-1]
#   # mode: one of tf.estimator.ModeKeys
#   # mappings: python list mapping ids in label to classes (using -1 for void class)

#   if mode is tf.estimator.ModeKeys.TRAIN:
#     mappings = params.training_problem_def['lids2cids']
#   elif mode is tf.estimator.ModeKeys.EVAL:
#     mappings = params.evaluation_problem_def['lids2cids']
#   elif mode is tf.estimator.ModeKeys.PREDICT:
#     mappings = params.inference_problem_def['lids2cids']
#   else:
#     assert False, f'mode {mode} not supported.'

#   return ids2image(label, _replacevoids(mappings))

def almost_equal(num1, num2, error=10**-3):
  return abs(num1-num2) <= error

def _replacevoids(mappings):
  # previous code replaced voids with max id + 1
  max_m = max(mappings)
  return [m if m!=-1 else max_m+1 for m in mappings]

# safe_div from tensorflow/python/ops/losses/losses_impl.py
def safe_div(num, den, name="safe_div"):
  """Computes a safe divide which returns 0 if the den is zero.
  Note that the function contains an additional conditional check that is
  necessary for avoiding situations where the loss is zero causing NaNs to
  creep into the gradient computation.
  Args:
    num: An arbitrary `Tensor`.
    den: `Tensor` whose shape matches `num` and whose values are
      assumed to be non-negative.
    name: An optional name for the returned op.
  Returns:
    The element-wise value of the num divided by the den.
  """
  return tf.where(tf.greater(den, 0),
                  tf.div(num,
                         tf.where(tf.equal(den, 0),
                                  tf.ones_like(den), den)),
                  tf.zeros_like(num),
                  name=name)

def print_metrics_from_confusion_matrix(cm, labels=None, printfile=None, printcmd=False, summary=False):
  # cm: numpy, 2D, square, np.int32 array, not containing NaNs
  # labels: python list of labels
  # printfile: file handler or None to print metrics to file
  # printcmd: True to print a summary of metrics to terminal
  # summary: if printfile is not None, prints only a summary of metrics to file
  cm = cm.astype(np.int32)
  # sanity checks
  assert isinstance(cm, np.ndarray), 'Confusion matrix must be numpy array.'
  cms = cm.shape
  assert all([cm.dtype==np.int32,
              cm.ndim==2,
              cms[0]==cms[1],
              not np.any(np.isnan(cm))]), (
                f"Check print_metrics_from_confusion_matrix input requirements. "
                f"Input has {cm.ndim} dims, is {cm.dtype}, has shape {cms[0]}x{cms[1]} "
                f"and may contain NaNs.")
  if not labels:
    labels = ['unknown']*cms[0]
  assert len(labels)==cms[0], (
    f"labels ({len(labels)}) must be enough for indexing confusion matrix ({cms[0]}x{cms[1]}).")
  #assert os.path.isfile(printfile), 'printfile is not a file.'
  
  # metric computations
  global_accuracy = np.trace(cm)/np.sum(cm)*100
  # np.sum(cm,1) can be 0 so some accuracies can be nan
  accuracies = np.diagonal(cm)/np.sum(cm,1)*100
  # denominator can be zero only if #TP=0 which gives nan, trick to avoid it
  inter = np.diagonal(cm)
  union = np.sum(cm,0)+np.sum(cm,1)-np.diagonal(cm)
  ious = inter/np.where(union>0,union,np.ones_like(union))*100
  notnan_mask = np.logical_not(np.isnan(accuracies))
  mean_accuracy = np.mean(accuracies[notnan_mask])
  mean_iou = np.mean(ious[notnan_mask])
  
  # reporting
  log_string = "\n"
  log_string += f"Global accuracy: {global_accuracy:5.2f}\n"
  log_string += "Per class accuracies (nans due to 0 #Trues) and ious (nans due to 0 #TPs):\n"
  for k,v in {l:(a,i,m) for l,a,i,m in zip(labels, accuracies, ious, notnan_mask)}.items():
    log_string += f"{k:<30s}  {v[0]:>5.2f}  {v[1]:>5.2f}  {'' if v[2] else '(ignored in averages)'}\n"
  log_string += f"Mean accuracy (ignoring nans): {mean_accuracy:5.2f}\n"
  log_string += f"Mean iou (ignoring accuracies' nans but including ious' 0s): {mean_iou:5.2f}\n"

  if printcmd:
    print(log_string)

  if printfile:
    if summary:
      printfile.write(log_string)
    else:
      print(f"{global_accuracy:>5.2f}",
            f"{mean_accuracy:>5.2f}",
            f"{mean_iou:>5.2f}",
            accuracies,
            ious,
            file=printfile)

def split_path(path):
  # filepath = <head>/<tail>
  # filepath = <head>/<root>.<ext[1:]>
  head, tail = os.path.split(path)
  root, ext = os.path.splitext(tail)
  return head, root, ext[1:]

def _validate_problem_config(config):
  # config is a json file that contains problem configuration
  #   it should contain at least the following fields:
  #     version: version of config json
  #     mappings: index:label id -> class id
  #     labels: index:class id -> label string
  #     palettes: index:class id -> RGB color
  #     idspalettes: index:class id -> label id (temporary field:
  #       can be infered from mappings)

  # check if all required keys are present
  mandatory_keys = ['version', 'mappings', 'labels', 'palettes', 'idspalettes']
  assert all([k in config.keys() for k in mandatory_keys]), (
    f"problem config json must also have {set(mandatory_keys)-set(config.keys())}")
  # check version
  assert config['version'] == 2.0

  ## type and value checking
  # check mappings validity
  assert all([isinstance(m, int) and m >= -1 for m in config['mappings']]), (
    'Only integers and >=-1 are allowed in mappings.')
  # check labels validity
  assert all([isinstance(l, str) for l in config['labels']]), (
    'Only string labels are allowed in labels.')
  # check palettes validity
  assert all([isinstance(c, int) and 0 <= c <= 255
              for p in config['palettes'] for c in p])
  
  ## length checking
  # Nclasses = count_non_i(config['mappings'], -1) + (1 if -1 in config['mappings'] else 0)
  # +1 since class ids start from 0 by convention, if -1 (void) exist another
  #   label, palette and idspalette are needed
  # TODO: check also problemXXTOproblemXX fields if they exist
  Nclasses = max(config['mappings']) + 1 + (1 if -1 in config['mappings'] else 0)
  lens = list(map(len, [config['labels'], config['palettes']]))
  assert all([l == Nclasses for l in lens]), (
    f"Lengths of labels and palettes ({lens}) are not compatible "
    f"with number of classes ({Nclasses}).")
  assert all([len(p) == 3 for p in config['palettes']])

def count_non_i(int_lst, i):
  # counts the number of integers not equal to i in the integer list int_lst
  
  # assertions
  assert isinstance(int_lst, list), 'int_lst is not a list.'
  assert all([isinstance(e, int) for e in int_lst]), 'Not integer int_list.'
  assert isinstance(i, int), 'Not integer i.'

  # implementation
  return len(list(filter(lambda k: k != i, int_lst)))

def cids2lids(cids, cids_unlabeled_id, lids_unlabeled_id):
  # cids: list of ids
  assert cids_unlabeled_id >= max(cids)
  lids = [lids_unlabeled_id]*(cids_unlabeled_id+1)
  for i, cid in enumerate(cids):
    lids[cid] = i
  return lids
