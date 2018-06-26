"""Example of how to use Semantic Segmentation system for prediction.
"""

import sys, os
import collections
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
plt.ion() # for expected waitforbuttonpress functionality
from PIL import Image
from system_factory import SemanticSegmentation
from input_cityscapes_mapillary_gtsdb.input_pipeline import predict_input as predict_fn
from model.model import model as model_fn
from utils.utils import SemanticSegmentationArguments, split_path


def main(argv):
  ssargs = SemanticSegmentationArguments()
  ssargs.add_predict_arguments()
  args = ssargs.parse_args(argv)

  _add_extra_args(args)

  # vars(args).items() returns (key,value) tuples from args.__dict__
  # and sorted uses first element of tuples to sort
  args_dict = collections.OrderedDict(sorted(vars(args).items()))
  print(args_dict)

  settings = args

  # move to system
  if args.results_dir and not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)

  system = SemanticSegmentation({'predict': predict_fn}, model_fn, settings)

  # TODO: 2D plotting not needed since estimator outputs one example
  # per step irrespective of settings.Nb
  Nb = 1 # not the same as settings.Nb
  Np = 2 # number of plots for each batch example

  if args.plotting:
    fig = plt.figure(0)
    axs = [[None for j in range(Np)] for i in range(Nb)]
    for i in range(Nb):
      for j in range(Np):
        axs[i][j] = fig.add_subplot(Np, Nb, Np*i+(j+1))
        axs[i][j].set_axis_off()
    fig.tight_layout()

  # color mappings for outputs
  if args.plotting:
    palettes = np.array(args.inference_problem_def['cids2colors'], dtype=np.uint8)
  if args.export_lids_images:
    idspalettes = np.array(args.inference_problem_def['cids2lids'], dtype=np.uint8)
    void_exists = -1 in args.inference_problem_def['cids2lids']
  if args.export_color_images:
    colorpalettes = np.array(args.inference_problem_def['cids2colors'], dtype=np.uint8)
    void_exists = -1 in args.inference_problem_def['cids2lids']

  start = datetime.now()
  start_for_total = datetime.now()
  for outputs in system.predict():
    # keep next line first in the loop for correct timings and don't print
    # anything else inside the loop for single line refresing
    sys.stdout.write(f'debug:time (valid for --Nb 1 only): {datetime.now()-start}\r')
    sys.stdout.flush()

    # unpack outputs
    # decs: 2D, np.int32, in [0,Nclasses-1]
    # rawimage: 3D, np.uint8
    # rawimagepath: 0D, bytes object, convert to str object (using str()) for using it
    decs = outputs['decisions']
    rawimage = outputs['rawimages']
    rawimagepath = str(outputs['rawimagespaths'])

    # live plotting
    if args.plotting:
      for i in range(Nb):
        axs[i][0].imshow(rawimage)
        axs[i][1].imshow(palettes[decs])
        # axs[i][2].imshow(np.amax(probs[i,...], 2), cmap='Greys_r')
      fig.tight_layout()
      plt.waitforbuttonpress(timeout=settings.timeout)

    # export label ids images
    if args.export_lids_images:
      result_ids = idspalettes[decs]
      # handle void in inference cids2lids
      if void_exists:
        assert np.all(np.not_equal(decs, max(system.settings.training_lids2cids))), (
            'void needs handling.')
      out_fname = os.path.join(args.results_dir, split_path(rawimagepath)[1] + '.png')
      # if you want to overwrite files comment next line
      assert not os.path.exists(out_fname), 'Output filename already exists.'
      Image.fromarray(result_ids).save(out_fname)

    if args.export_color_images:
      # if current_media.ftype=='video':
      #   assert False, 'Not implemented. The selected media folder contains video files.'
      result_col = colorpalettes[decs]
      out_fname = os.path.join(args.results_dir, split_path(rawimagepath)[1] + '_resultCol.png')
      # if you want to overwrite files comment next line
      assert not os.path.exists(out_fname), 'Output filename already exists.'
      Image.fromarray(result_col).save(out_fname)

      #Image.fromarray(result_col, mode="RGB").save(out_path)
    # if args.export_probs_pickle:
    #   # decs: 3D numpy, np.float32, in [0,1]
    #   print('\n\nWARNING: code block not tested.\n\n')
    #   probs = preds['probabilities']
    #   print('debug:probs:', probs.shape, probs.dtype)
    #   out_path = join(args.results_dir, current_media.media.fname + '_resultProbs' + '.p')
    #   with open(out_path, 'wb') as f:
    #     pickle.dump(probs, f)

    # keep next line last for correct timings
    start = datetime.now()

  print('\nTotal time:', datetime.now()-start_for_total)


def _add_extra_args(args):
  # disable regularizer and set batch_norm_decay to random value for with... to work
  args.batch_norm_istraining = False
  args.regularization_weight = 0.0
  args.batch_norm_decay = 1.0

  # prediction keys for SemanticSegmentation.predict(), predictions are defined in model.py
  args.predict_keys = ['decisions', 'rawimages', 'rawimagespaths']

if __name__ == '__main__':
  main(sys.argv[1:])
