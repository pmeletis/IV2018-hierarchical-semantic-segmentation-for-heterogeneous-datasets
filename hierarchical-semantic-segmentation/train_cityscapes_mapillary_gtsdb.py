
import sys
import collections
import os

import tensorflow as tf
# verbosity is set at system_factory so it is universal
# tf.logging.set_verbosity('DEBUG')

from system_factory import SemanticSegmentation
from input_cityscapes_mapillary_gtsdb.input_pipeline import train_input as train_fn
# from input_mapillary.input_pipeline import train_input as train_fn
from model.model import model as model_fn
from utils.utils import SemanticSegmentationArguments
from utils.util_zip import zipit
from os.path import split, join, realpath


def main(argv):
  ssargs = SemanticSegmentationArguments()
  ssargs.add_train_arguments()
  args = ssargs.parse_args(argv)

  _add_extra_args(args)

  # vars(args).items() returns (key,value) tuples from args.__dict__
  # and sorted uses first element of tuples to sort
  args_dict = collections.OrderedDict(sorted(vars(args).items()))
  print(args_dict)

  settings = args

  system = SemanticSegmentation({'train': train_fn}, model_fn, settings)

  zipit(split(realpath(__file__))[0], join(system.settings.log_dir, 'all_code.zip'))

  system.train()

def _add_extra_args(args):
  # extra args for train
  # since mapillary is the biggest dataset train according to its epochs
  args.num_examples = int(args.Ntrain_mapil *
                          args.height_network//args.height_feature_extractor *
                          args.width_network//args.width_feature_extractor) # per epoch
  args.num_batches_per_epoch = int(args.num_examples / args.Nb_mapil)
  args.num_training_steps = int(args.num_batches_per_epoch * args.Ne) # per training
  if args.learning_rate_schedule == 'piecewise_constant':
    # best: 8, 7, 2
    args.num_decay_steps = 2
    args.lr_boundaries = [8*args.num_batches_per_epoch] # <=8 (if >8 starts to diverge (for bn_decay=0.9, earlier for 0.99, kok))
    for ne in [7]: # 6
      args.lr_boundaries.append(ne * args.num_batches_per_epoch + args.lr_boundaries[-1])
    #print('debug:lr_boundaries:', type(args.lr_boundaries[1]))
    assert len(args.lr_boundaries)==args.num_decay_steps
    args.lr_values = [args.learning_rate_initial * args.learning_rate_decay**i for i in range(args.num_decay_steps+1)]
    print('debug:lr_values,lr_boundaries,num_training_steps:', args.lr_values, args.lr_boundaries, args.num_training_steps)
  # elif args.learning_rate_schedule=='exponential_decay':
  #   assert False, 'Not tested...'
  #   args.Ne_per_decay = 12
  #   args.num_batches_per_decay = int(args.num_batches_per_epoch * args.Ne_per_decay)
  #   args.num_decay_steps = int(args.Ne // args.Ne_per_decay) # results num_decay_steps + 1 different learning rates
  #   #args.staircase = False
  #  #elif args.learning_rate_schedule=='polynomial_decay':
  #   #args.end_learning_rate = 0.001
  #   #args.Ne_per_decay = args.Ne - 2 # 2 epochs for training with end_learning_rate
  #   #args.num_batches_per_decay = int(args.num_batches_per_epoch * args.Ne_per_decay)
  #   #args.power = 0.6
  else:
    assert False, 'No such learning rate schedule.'

  # train batch normalization layers
  args.batch_norm_istraining = True

  # loss regularization
  args.lamda = [[1.0], [0.1, 0.1, 0.1], [0.1]]

if __name__ == '__main__':
  main(sys.argv[1:])
