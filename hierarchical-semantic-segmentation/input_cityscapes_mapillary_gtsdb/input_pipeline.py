import tensorflow as tf
import sys, glob
from os.path import join, split, realpath
import functools
sys.path.append(split(realpath(__file__))[0])
from preprocess_augmentation_1 import preprocess_train
from PIL import Image
import numpy as np

# _TRAIN_SIZE = (512, 706)

def train_parse_prepare_preprocess_mapillary(example, params):
  _TRAIN_SIZE = (params.height_feature_extractor, params.width_feature_extractor)
  # example: a serialized tf example
  # proimage: 3D, tf.float32, _TRAIN_SIZE
  # prolabel: 2D, tf.int32, _TRAIN_SIZE

  ## parse
  keys_to_features = {
    'image/encoded': tf.FixedLenFeature((), tf.string, default_value=None),
    'image/format':  tf.FixedLenFeature((), tf.string, default_value=b'jpeg'),
    'image/dtype':   tf.FixedLenFeature((), tf.string, default_value=b'uint8'),
    'image/shape':   tf.FixedLenFeature((3), tf.int64, default_value=None),
    'image/path':    tf.FixedLenFeature((), tf.string, default_value=None),
    'label/encoded': tf.FixedLenFeature((), tf.string, default_value=None),
    'label/format':  tf.FixedLenFeature((), tf.string, default_value=b'png'),
    'label/dtype':   tf.FixedLenFeature((), tf.string, default_value=b'uint8'),
    'label/shape':   tf.FixedLenFeature((3), tf.int64, default_value=None),
    'label/path':    tf.FixedLenFeature((), tf.string, default_value=None)}

  features = tf.parse_single_example(example, keys_to_features)

  image = tf.image.decode_jpeg(features['image/encoded'])
  label = tf.image.decode_png(features['label/encoded'])
  # label = label[..., 0]
  im_path = features['image/path']
  la_path = features['label/path']

  ## prepare
  # TF internal bug: resize_images semantics don't correspond to uint8 images
  # it needs an input of [0,1] to work properly
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  # label = tf.gather(tf.to_float(lids2cids), tf.to_int32(label))
  label = tf.gather(tf.cast(params.training_lids2cids_mapil, tf.uint8), tf.to_int32(label))
  print('debug: prepare:', image, label)

  ## preprocess
  ## rectify sizes before random cropping
  # make the smaller dimension at least as large as the respective train dimension
  def true_fn(image, label):
    # the size which is smaller than the respective average size will have bigger scale
    scales = _TRAIN_SIZE / tf.shape(image)[:2]
    new_size = tf.to_int32(tf.reduce_max(scales) * tf.saturate_cast(tf.shape(image)[:2], tf.float64))
    # + 1 to new_size just in case to_int32 does floor rounding
    image = tf.image.resize_images(image, new_size + 1)
    # image has to be 3D
    label = tf.image.resize_images(label, new_size + 1,
                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return image, label

  def false_fn(image, label):
    im_spatial_shape = tf.shape(image)[:2]
    mult = tf.constant(_TRAIN_SIZE)[tf.argmin(im_spatial_shape)] / tf.reduce_min(im_spatial_shape)
    new_spatial_size = tf.to_int32(tf.saturate_cast(im_spatial_shape, tf.float64) * mult)
    # + 1 to new_size just in case to_int32 does floor rounding
    image = tf.image.resize_images(image, new_spatial_size + 1)
    # image has to be 3D
    label = tf.image.resize_images(label, new_spatial_size + 1,
                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return image, label

  # if any dim is smaller than the respective _TRAIN_SIZE rectify it
  image, label = tf.cond(tf.reduce_any(tf.less(tf.shape(image)[:2], _TRAIN_SIZE)),
                         true_fn=lambda: true_fn(image, label),
                         false_fn=lambda: false_fn(image, label))

  # one more time same cond since false_fn can create smaller images
  image, label = tf.cond(tf.reduce_any(tf.less(tf.shape(image)[:2], _TRAIN_SIZE)),
                         true_fn=lambda: true_fn(image, label),
                         false_fn=lambda: (image, label))

  # random crop
  # trick to randomly crop the same area from image and label
  # print('debug: before random crop:', image, label)
  # image = tf.Print(image, [tf.reduce_min(image), tf.reduce_max(image)], message='image min, max: ')
  image = tf.image.convert_image_dtype(image, tf.uint8, saturate=True) # so it can be concated
  combined = tf.concat([image, label], 2)
  combined = tf.random_crop(combined, _TRAIN_SIZE + tuple([4]))
  image = combined[..., :3]
  label = combined[..., 3]

  image = tf.image.convert_image_dtype(image, tf.float32)
  label = tf.to_int32(label)
  print('debug: crop:', image, label)

  # after this point image and label has _TRAIN_SIZE size

  # center input to [-1,1] is equivalent to assuming mean of 0.5
  mean = 0.5
  image = (image - mean)/mean
  proimage = image
  prolabel = label

  # proimage, prolabel = preprocess_train(image, label, params)

  return proimage, prolabel, im_path, la_path

def train_parse_prepare_preprocess_cityscapes(example, params):
  _TRAIN_SIZE = (params.height_feature_extractor, params.width_feature_extractor)
  ## parse
  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value=b'png'),
      'image/path': tf.FixedLenFeature((), tf.string, default_value=''),
      'label/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'label/format': tf.FixedLenFeature((), tf.string, default_value=b'png'),
      'label/path': tf.FixedLenFeature((), tf.string, default_value=''),
      'height': tf.FixedLenFeature((), tf.int64, default_value=1024),
      'width': tf.FixedLenFeature((), tf.int64, default_value=2048)
      }

  features = tf.parse_single_example(example, keys_to_features)

  image = tf.image.decode_png(features['image/encoded'])
  label = tf.image.decode_png(features['label/encoded'])
  # label = label[..., 0]
  im_path = features['image/path']
  la_path = features['label/path']

  ## prepare
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  label = tf.gather(tf.cast(params.training_lids2cids_citys, tf.uint8), tf.to_int32(label))
  print('debug: prepare:', image, label)

  ## preprocess
  image = tf.image.resize_images(image, (512, 1024))
  label = tf.image.resize_images(label, (512, 1024), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # random crop to _TRAIN_SIZE
  # trick to randomly crop the same area from image and label
  image = tf.image.convert_image_dtype(image, tf.uint8, saturate=True) # so it can be concated
  combined = tf.concat([image, label], 2)
  combined = tf.random_crop(combined, _TRAIN_SIZE + tuple([4]))
  image = combined[..., :3]
  label = combined[..., 3]

  image = tf.image.convert_image_dtype(image, tf.float32)
  label = tf.to_int32(label)
  print(image, label)

  # mean = 0.5
  # image = (image - mean)/mean
  # proimage = image
  # prolabel = label

  proimage, prolabel = preprocess_train(image, label, params)

  return proimage, prolabel, im_path, la_path

def train_parse_prepare_preprocess_gtsdb(example, params):
  _TRAIN_SIZE = (params.height_feature_extractor, params.width_feature_extractor)
  # proimage: 3D, tf.float32, _TRAIN_SIZE
  # prolabel: 2D, tf.int32, _TRAIN_SIZE

  keys_to_features = {
    'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
    'image/format': tf.FixedLenFeature((), tf.string, default_value=b'png'),
    'label/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
    'label/format': tf.FixedLenFeature((), tf.string, default_value=b'png'),
    'height': tf.FixedLenFeature((), tf.int64, default_value=800),
    'width': tf.FixedLenFeature((), tf.int64, default_value=1360)}

  features = tf.parse_single_example(example, keys_to_features)
  
  image = tf.image.decode_png(features['image/encoded'])
  label = tf.image.decode_png(features['label/encoded'], dtype=tf.uint16)

  ## prepare
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  label = tf.gather(tf.cast(params.training_lids2cids_gtsdb, tf.uint16), tf.to_int32(label))
  print('debug: prepare:', image, label)

  ## preprocess
  image = tf.image.resize_images(image, (512, 850))
  label = tf.image.resize_images(label, (512, 850), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # random crop to _TRAIN_SIZE
  # trick to randomly crop the same area from image and label
  image = tf.image.convert_image_dtype(image, tf.uint16, saturate=True) # so it can be concated
  combined = tf.concat([image, label], 2)
  combined = tf.random_crop(combined, _TRAIN_SIZE + tuple([4]))
  image = combined[..., :3]
  label = combined[..., 3]

  image = tf.image.convert_image_dtype(image, tf.float32)
  label = tf.to_int32(label)
  print(image, label)

  # mean = 0.5
  # image = (image - mean)/mean
  # proimage = image
  # prolabel = label

  proimage, prolabel = preprocess_train(image, label, params)

  return proimage, prolabel, tf.constant(''), tf.constant('')

def create_dataset(ppp, tfrecords_path, Nb, params):
  dataset = tf.data.TFRecordDataset(tfrecords_path)
  dataset = dataset.repeat()
  ppp = functools.partial(ppp, params=params)
  dataset = dataset.map(ppp, num_parallel_calls=8)
  dataset = dataset.batch(Nb)
  # shuffling at the end of the pipeline as it is the only way for now for buffering
  # buffering is needed to speedup the pipeline to the limits of CPU
  dataset = dataset.shuffle(buffer_size=40)
  return dataset

def _concatenate_datasets(dataset1, dataset2, dataset3):
  datasets = tf.data.Dataset.zip((dataset1, dataset2, dataset3))
  # print(datasets)
  def cc(tensor_list):
    return tf.concat(tensor_list, 0)
  datasets = datasets.map(
      lambda f, s, t: (
          cc([f[0], s[0], t[0]]),
          cc([f[1], s[1], t[1]]),
          cc([f[2], s[2], t[2]]),
          cc([f[3], s[3], t[3]])),
      num_parallel_calls=8)
  return datasets

def train_input(config, params):
  del config
  dataset_mapil = create_dataset(train_parse_prepare_preprocess_mapillary,
                                 params.tfrecords_path_mapil,
                                 params.Nb_mapil,
                                 params)
  dataset_citys = create_dataset(train_parse_prepare_preprocess_cityscapes,
                                 params.tfrecords_path_citys,
                                 params.Nb_citys,
                                 params)
  dataset_gtsdb = create_dataset(train_parse_prepare_preprocess_gtsdb,
                                 params.tfrecords_path_gtsdb,
                                 params.Nb_gtsdb,
                                 params)

  print(dataset_mapil)
  print(dataset_citys)
  print(dataset_gtsdb)

  datasets = _concatenate_datasets(dataset_citys, dataset_mapil, dataset_gtsdb)
  # print(datasets)
  iterator = datasets.make_one_shot_iterator()

  values = iterator.get_next()

  # merge already known batch dimension for _unpad4d to work
  # values[0].set_shape((params.Nb, None, None, 3))
  # values[1].set_shape((params.Nb, None, None))

  # Note: for now estimator code doesn't support lists of raw{images, labels},
  # thus send just the first element to the summaries
  features = {
      # 'rawimages': _unpad4d_to_list(values[0], values[2])[0][tf.newaxis, ...],
      'rawimages': tf.zeros_like(values[0]),
      'proimages': values[0],
      'rawimagespaths': values[2],
      'rawlabelspaths': values[3],
      }
  labels = {
      # 'rawlabels': _unpad4d_to_list(values[1][..., tf.newaxis], values[3])[0][..., 0][tf.newaxis, ...],
      'rawlabels': tf.zeros_like(values[1]),
      'prolabels': values[1],
      }

  return features, labels

def _replacevoids(mappings):
  # replace voids encoded by convention with -1 with max cid + 1
  void_cid = max(mappings) + 1
  return [m if m != -1 else void_cid for m in mappings]

def evaluate_parse_prepare_preprocess_mapillary(example, params):
  # _TRAIN_SIZE = (512, 706)
  # example: a serialized tf example
  # proimage: 3D, tf.float32, _TRAIN_SIZE
  # prolabel: 2D, tf.int32, _TRAIN_SIZE

  ## parse
  keys_to_features = {
    'image/encoded': tf.FixedLenFeature((), tf.string, default_value=None),
    'image/format':  tf.FixedLenFeature((), tf.string, default_value=b'jpeg'),
    'image/dtype':   tf.FixedLenFeature((), tf.string, default_value=b'uint8'),
    'image/shape':   tf.FixedLenFeature((3), tf.int64, default_value=None),
    'image/path':    tf.FixedLenFeature((), tf.string, default_value=None),
    'label/encoded': tf.FixedLenFeature((), tf.string, default_value=None),
    'label/format':  tf.FixedLenFeature((), tf.string, default_value=b'png'),
    'label/dtype':   tf.FixedLenFeature((), tf.string, default_value=b'uint8'),
    'label/shape':   tf.FixedLenFeature((3), tf.int64, default_value=None),
    'label/path':    tf.FixedLenFeature((), tf.string, default_value=None)}

  features = tf.parse_single_example(example, keys_to_features)

  image = tf.image.decode_jpeg(features['image/encoded'])
  label = tf.image.decode_png(features['label/encoded'])
  label = label[..., 0]

  im_path = features['image/path']
  la_path = features['label/path']

  ## prepare
  # TF internal bug: resize_images semantics don't correspond to uint8 images
  # it needs an input of [0,1] to work properly
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  label = tf.gather(tf.cast(_replacevoids(params.evaluation_problem_def['lids2cids']),
                            tf.int32),
                    tf.to_int32(label))
  print(image, label)

  ## preprocess
  image = tf.image.resize_images(image, (params.height_network, params.width_network))
  label = tf.image.resize_images(label[..., tf.newaxis],
                                 (params.height_network, params.width_network),
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)[..., 0]

  # center input to [-1,1] is equivalent to assuming mean of 0.5
  mean = 0.5
  image = (image - mean)/mean
  proimage = image
  prolabel = label

  return proimage, prolabel, im_path, la_path

def evaluate_input_mapillary(config, params):
  del config
  dataset = tf.data.TFRecordDataset(params.tfrecords_path)
  ppp_mapillary = functools.partial(evaluate_parse_prepare_preprocess_mapillary, params=params)
  dataset = dataset.map(ppp_mapillary, num_parallel_calls=8)
  dataset = dataset.batch(params.Nb)
  # shuffling at the end of the pipeline as it is the only way for now for buffering
  # buffering is needed to speedup the pipeline to the limits of CPU
  dataset = dataset.shuffle(buffer_size=40)

  iterator = dataset.make_one_shot_iterator()

  values = iterator.get_next()

  # merge with known shape as requested by model
  values[0].set_shape([None, None, None, 3])

  features = {
      'rawimages': tf.zeros_like(values[0]),
      'proimages': values[0],
      'rawimagespaths': values[2],
      'rawlabelspaths': values[3],
      }
  labels = {
      'rawlabels': tf.zeros_like(values[1]),
      'prolabels': values[1],
      }

  return features, labels

def evaluate_parse_prepare_preprocess_cityscapes(example, params):
  # _TRAIN_SIZE = (512, 706)
  # example: a serialized tf example
  # proimage: 3D, tf.float32, _TRAIN_SIZE
  # prolabel: 2D, tf.int32, _TRAIN_SIZE

  ## parse
  keys_to_features = {
    'image/encoded': tf.FixedLenFeature((), tf.string, default_value=None),
    'image/format':  tf.FixedLenFeature((), tf.string, default_value=b'jpeg'),
    'image/dtype':   tf.FixedLenFeature((), tf.string, default_value=b'uint8'),
    'image/shape':   tf.FixedLenFeature((3), tf.int64, default_value=None),
    'image/path':    tf.FixedLenFeature((), tf.string, default_value=None),
    'label/encoded': tf.FixedLenFeature((), tf.string, default_value=None),
    'label/format':  tf.FixedLenFeature((), tf.string, default_value=b'png'),
    'label/dtype':   tf.FixedLenFeature((), tf.string, default_value=b'uint8'),
    'label/shape':   tf.FixedLenFeature((3), tf.int64, default_value=None),
    'label/path':    tf.FixedLenFeature((), tf.string, default_value=None)}

  features = tf.parse_single_example(example, keys_to_features)

  image = tf.image.decode_png(features['image/encoded'])
  label = tf.image.decode_png(features['label/encoded'])
  label = label[..., 0]

  im_path = features['image/path']
  la_path = features['label/path']

  ## prepare
  # TF internal bug: resize_images semantics don't correspond to uint8 images
  # it needs an input of [0,1] to work properly
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  label = tf.gather(tf.cast(_replacevoids(params.evaluation_problem_def['lids2cids']),
                            tf.int32),
                    tf.to_int32(label))
  print(image, label)

  ## preprocess
  image = tf.image.resize_images(image, (params.height_network, params.width_network))
  label = tf.image.resize_images(label[..., tf.newaxis],
                                 (params.height_network, params.width_network),
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)[..., 0]

  # center input to [-1,1] is equivalent to assuming mean of 0.5
  mean = 0.5
  image = (image - mean)/mean
  proimage = image
  prolabel = label

  return proimage, prolabel, im_path, la_path

def evaluate_input_cityscapes(config, params):
  del config
  dataset = tf.data.TFRecordDataset(params.tfrecords_path)
  ppp_cityscapes = functools.partial(evaluate_parse_prepare_preprocess_cityscapes, params=params)
  dataset = dataset.map(ppp_cityscapes, num_parallel_calls=8)
  dataset = dataset.batch(params.Nb)
  # shuffling at the end of the pipeline as it is the only way for now for buffering
  # buffering is needed to speedup the pipeline to the limits of CPU
  dataset = dataset.shuffle(buffer_size=40)

  iterator = dataset.make_one_shot_iterator()

  values = iterator.get_next()

  # merge with known shape as requested by model
  values[0].set_shape([None, None, None, 3])

  features = {
      'rawimages': tf.zeros_like(values[0]),
      'proimages': values[0],
      'rawimagespaths': values[2],
      'rawlabelspaths': values[3],
      }
  labels = {
      'rawlabels': tf.zeros_like(values[1]),
      'prolabels': values[1],
      }

  return features, labels

def evaluate_parse_prepare_preprocess_cityscapes_extended(example, params):
  # _TRAIN_SIZE = (512, 706)
  # example: a serialized tf example
  # proimage: 3D, tf.float32, _TRAIN_SIZE
  # prolabel: 2D, tf.int32, _TRAIN_SIZE

  ## parse
  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value=b'png'),
      'label/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'label/format': tf.FixedLenFeature((), tf.string, default_value=b'png'),
      'height': tf.FixedLenFeature((), tf.int64, default_value=1024),
      'width': tf.FixedLenFeature((), tf.int64, default_value=2048),
      }

  features = tf.parse_single_example(example, keys_to_features)

  image = tf.image.decode_png(features['image/encoded'])
  label = tf.image.decode_png(features['label/encoded'], dtype=tf.uint16)
  label = label[..., 0]

  # im_path = features['image/path']
  # la_path = features['label/path']

  ## prepare
  # TF internal bug: resize_images semantics don't correspond to uint8 images
  # it needs an input of [0,1] to work properly
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  label = tf.gather(tf.cast(_replacevoids(params.evaluation_problem_def['lids2cids']),
                            tf.int32),
                    tf.to_int32(label))
  print(image, label)

  ## preprocess
  image = tf.image.resize_images(image, (params.height_network, params.width_network))
  label = tf.image.resize_images(label[..., tf.newaxis],
                                 (params.height_network, params.width_network),
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)[..., 0]

  # center input to [-1,1] is equivalent to assuming mean of 0.5
  mean = 0.5
  image = (image - mean)/mean
  proimage = image
  prolabel = label

  return proimage, prolabel, tf.constant(''), tf.constant('')

def evaluate_input_cityscapes_extended(config, params):
  del config
  dataset = tf.data.TFRecordDataset(params.tfrecords_path)
  ppp = functools.partial(evaluate_parse_prepare_preprocess_cityscapes_extended, params=params)
  dataset = dataset.map(ppp, num_parallel_calls=8)
  dataset = dataset.batch(params.Nb)
  # shuffling at the end of the pipeline as it is the only way for now for buffering
  # buffering is needed to speedup the pipeline to the limits of CPU
  dataset = dataset.shuffle(buffer_size=40)

  iterator = dataset.make_one_shot_iterator()

  values = iterator.get_next()

  # merge with known shape as requested by model
  values[0].set_shape([None, None, None, 3])

  features = {
      'rawimages': tf.zeros_like(values[0]),
      'proimages': values[0],
      'rawimagespaths': values[2],
      'rawlabelspaths': values[3],
      }
  labels = {
      'rawlabels': tf.zeros_like(values[1]),
      'prolabels': values[1],
      }

  return features, labels

def evaluate_parse_prepare_preprocess_gtsdb(example, params):
  # _TRAIN_SIZE = (512, 706)
  # example: a serialized tf example
  # proimage: 3D, tf.float32, _TRAIN_SIZE
  # prolabel: 2D, tf.int32, _TRAIN_SIZE

  ## parse
  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value=b'png'),
      'label/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'label/format': tf.FixedLenFeature((), tf.string, default_value=b'png'),
      'height': tf.FixedLenFeature((), tf.int64, default_value=800),
      'width': tf.FixedLenFeature((), tf.int64, default_value=1360),
      }

  features = tf.parse_single_example(example, keys_to_features)

  image = tf.image.decode_png(features['image/encoded'])
  label = tf.image.decode_png(features['label/encoded'], dtype=tf.uint16)
  label = label[..., 0]

  # im_path = features['image/path']
  # la_path = features['label/path']

  ## prepare
  # TF internal bug: resize_images semantics don't correspond to uint8 images
  # it needs an input of [0,1] to work properly
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  label = tf.gather(tf.cast(_replacevoids(params.evaluation_problem_def['lids2cids']),
                            tf.int32),
                    tf.to_int32(label))
  print(image, label)

  ## preprocess
  image = tf.image.resize_images(image, (params.height_network, params.width_network))
  label = tf.image.resize_images(label[..., tf.newaxis],
                                 (params.height_network, params.width_network),
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)[..., 0]

  # center input to [-1,1] is equivalent to assuming mean of 0.5
  mean = 0.5
  image = (image - mean)/mean
  proimage = image
  prolabel = label

  return proimage, prolabel, tf.constant(''), tf.constant('')

def evaluate_input_gtsdb(config, params):
  del config
  dataset = tf.data.TFRecordDataset(params.tfrecords_path)
  ppp = functools.partial(evaluate_parse_prepare_preprocess_gtsdb, params=params)
  dataset = dataset.map(ppp, num_parallel_calls=16)
  dataset = dataset.batch(params.Nb)
  # shuffling at the end of the pipeline as it is the only way for now for buffering
  # buffering is needed to speedup the pipeline to the limits of CPU
  dataset = dataset.shuffle(buffer_size=40)

  iterator = dataset.make_one_shot_iterator()

  values = iterator.get_next()

  # merge with known shape as requested by model
  values[0].set_shape([None, None, None, 3])

  features = {
      'rawimages': tf.zeros_like(values[0]),
      'proimages': values[0],
      'rawimagespaths': values[2],
      'rawlabelspaths': values[3],
      }
  labels = {
      'rawlabels': tf.zeros_like(values[1]),
      'prolabels': values[1],
      }

  return features, labels

def predict_image_generator(params):
  SUPPORTED_EXTENSIONS = ['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG', 'ppm', 'PPM']
  fnames = []
  for se in SUPPORTED_EXTENSIONS:
    fnames.extend(glob.glob(join(params.predict_dir, '*.' + se), recursive=True))

  for im_fname in fnames:
    im = Image.open(im_fname)
    # next line is time consuming (can take up to 400ms for im of 2 MPixels)
    # im_array = np.array(im)
    # yield im_array, im_fname.encode('utf-8'), im_array.shape[0], im_array.shape[1]
    yield im, im_fname.encode('utf-8'), im.height, im.width

def predict_prepare_and_preprocess(im, params):
  im.set_shape((None, None, 3))
  im = tf.image.convert_image_dtype(im, tf.float32)
  print('debug: predict_prepare_and_preprocess:', im)
  im = tf.image.resize_images(im, [params.height_network, params.width_network])
  # center input to [-1,1] is equivalent to assuming mean of 0.5
  mean = 0.5
  proim = (im - mean)/mean

  return proim

def predict_input(config, params):
  dataset = tf.data.Dataset.from_generator(lambda: predict_image_generator(params),
                                           output_types=(tf.uint8, tf.string, tf.int32, tf.int32))
  dataset = dataset.map(lambda im, im_path, height, width: (im_path, im, predict_prepare_and_preprocess(im, params)), num_parallel_calls=16)
  dataset = dataset.batch(params.Nb)
  dataset = dataset.prefetch(params.Nb*20)
  iterator = dataset.make_one_shot_iterator()
  values = iterator.get_next()

  features = {
      'rawimages': values[1],
      'proimages': values[2],
      'rawimagespaths': values[0],
      }

  return features
