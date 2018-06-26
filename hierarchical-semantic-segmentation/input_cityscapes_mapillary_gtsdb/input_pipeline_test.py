from datetime import datetime
import tensorflow as tf
from input_pipeline import train_input

_PATH_MAPIL = ('/media/panos/data/datasets/mapillary'
              '/mapillary-vistas-dataset_public_v1.0/tfrecords/train.tfrecord')
_PATH_CITYS = ('/media/panos/data/datasets/cityscapes'
              '/tfrecords/trainFine_v4.tfrecords')

def train_input_test():
  # !!! IMPORTANT !!!
  # cannot use e.g. features['rawimages'].eval() and features['rawlabels'].eval()
  # because every eval causes the required nodes to be computed again and each eval
  # reads new examples...
  sess = tf.InteractiveSession()
  class params(object):
    height_feature_extractor = 512
    width_feature_extractor = 706
    Nb_mapil = 2
    Nb_citys = 1
    tfrecords_path_mapil = _PATH_MAPIL
    tfrecords_path_citys = _PATH_CITYS
    training_problem_def_mapil = {'lids2cids':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, -1]}
    training_lids2cids_mapil = training_problem_def_mapil['lids2cids']
    training_problem_def_citys = {'lids2cids':[-1,-1,-1,-1,-1,-1,-1, 0, 1,-1,-1, 2, 3, 4,-1,-1,-1, 5,-1, 6, 7, 8, 9,10,11,12,13,14,15,-1,-1,16,17,18]}
    training_lids2cids_citys = training_problem_def_citys['lids2cids']
    Ntrain_mapil = 18000
    Ntrain_citys = 2975
  with tf.device('/cpu:0'):
    features, labels = train_input(None, params)
  print(features, labels)
  for i in range(params.Ntrain_citys//params.Nb_citys):
    start = datetime.now()
    fe, la = sess.run((features, labels))
    print(
        fe['rawimages'].shape,
        la['rawlabels'].shape,
        fe['proimages'].shape,
        la['prolabels'].shape,
        fe['rawimagespaths'],
        fe['rawlabelspaths'],
        sep='\n',
        end='\n--------------------------------------\n',
        )
    print(i, datetime.now()-start)

def main():
  train_input_test()
  return

if __name__ == '__main__':
  main()
