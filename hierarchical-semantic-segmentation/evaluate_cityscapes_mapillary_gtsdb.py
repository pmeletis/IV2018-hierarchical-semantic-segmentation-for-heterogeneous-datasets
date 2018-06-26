
import sys
import collections
import os

import tensorflow as tf
# verbosity is set at system_factory so it is universal
# tf.logging.set_verbosity('DEBUG')

# defines how metrics will be printed to file
import numpy as np
np.set_printoptions(formatter={'float': '{:>5.2f}'.format}, nanstr=u'nan', linewidth=10000)

from system_factory import SemanticSegmentation
from input_cityscapes_mapillary_gtsdb.input_pipeline import evaluate_input_gtsdb as eval_fn
from model.model import model as model_fn
from utils.utils import (SemanticSegmentationArguments, print_metrics_from_confusion_matrix,
                         split_path)
# from utils.util_zip import zipit
from os.path import split, join, realpath
import pickle


def main(argv):
  ssargs = SemanticSegmentationArguments()
  ssargs.add_evaluate_arguments()
  args = ssargs.parse_args(argv)

  _add_extra_args(args)

  # vars(args).items() returns (key,value) tuples from args.__dict__
  # and sorted uses first element of tuples to sort
  args_dict = collections.OrderedDict(sorted(vars(args).items()))
  print(args_dict)

  settings = args

  # zipit(split(realpath(__file__))[0], join(system.settings.log_dir, 'all_code.zip'))

  labels = settings.evaluation_problem_def['cids2labels']
  void_exists = -1 in settings.evaluation_problem_def['lids2cids']
  labels = labels[:-1] if void_exists else labels

  system = SemanticSegmentation({'eval': eval_fn}, model_fn, settings)

  all_metrics = system.evaluate()

  # print full metrics to readable file
  mr_filename = os.path.join(system.settings.eval_res_dir, 'all_metrics.txt')
  with open(mr_filename, 'w') as f:
    for metrics in all_metrics:
      print(f"{metrics['global_step']:>06} ", end='', file=f)
      print_metrics_from_confusion_matrix(metrics['confusion_matrix'], labels, printfile=f)

  # save raw metrics to pickle for future reference
  # TODO: maybe move to system
  m_filename = os.path.join(system.settings.eval_res_dir, 'all_metrics.p')
  with open(m_filename, 'wb') as f:
    pickle.dump(all_metrics, f)

def _add_extra_args(args):
  # number of examples per epoch
  args.num_examples = int(args.Neval *
                          args.height_network//args.height_feature_extractor *
                          args.width_network//args.width_feature_extractor)
  args.num_batches_per_epoch = int(args.num_examples / args.Nb)
  args.num_eval_steps = int(args.num_batches_per_epoch * 1) # 1 epoch

  # disable regularizer and set batch_norm_decay to random value
  # temp solution so as with blocks to work
  args.batch_norm_istraining = False
  args.regularization_weight = 0.0
  args.batch_norm_decay = 1.0

  # force disable XLA, since there is an internal TF error till at least r1.4
  # TODO: remove this when error is fixed
  args.enable_xla = False

  # temporary
  args.loss_regularization = [10., 10., 10.]

if __name__ == '__main__':
  main(sys.argv[1:])
