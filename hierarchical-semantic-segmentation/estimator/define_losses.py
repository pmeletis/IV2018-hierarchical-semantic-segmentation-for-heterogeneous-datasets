"""Losses for Semantic Segmentation model supporting void labels.
"""

import tensorflow as tf


def define_losses(mode, config, params, predictions, labels):
  """
  specific to Cityscapes + Mapillary + GTSDB training

  labels: Nb x hf x wf, tf.int32, with elements in [0,Nc-1]
          labels[0]: cityscapes, labels[1:3]: mapillary, labels[3]: gtsdb
          encoded using per-dataset problem definition: [27+1, 65+1, 43+1]
  predictions: contains logits and decisions (at least)
  logits: [l1_logits, l2_logits, l3_logits] each of them: Nb x hf x wf x Nc, tf.float32
  by convention, if void label exists it has index Nc-1 and is ignored in the losses
  for now only mapillary dataset
  l1: 53+1 classes
  l2: [10+1, 3+1, 2+1] classes for classes 20 and 47 respectively
  l3: 43+1 classes
  """
  # TODO: find last classId statically (using mappings)
  # TODO: differentiate according to mode 

  del config

  if mode == tf.estimator.ModeKeys.EVAL:
    return {'total': tf.constant(0.),
            'segmentation': [[tf.constant(0.)],
                             [tf.constant(0.), tf.constant(0.), tf.constant(0.)],
                             [tf.constant(0.)]],
            'regularization': tf.constant(0.)}

  l1_logits, l2_logits, l3_logits = predictions['logits']
  l1_decs, l2_decs, l3_decs = predictions['decisions']
  lamda = params.lamda

  def _xentr_loss(labels, logits, weights, reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS, loss_collection=None):
    return tf.losses.sparse_softmax_cross_entropy(
        labels,
        logits,
        weights=weights,
        reduction=reduction,
        loss_collection=loss_collection)

  ## losses are organized per classifier
  # 1) labels per dataset are converted to common class ids (per classifier since they need to
  # start from 0 and be continuous)
  # 2) weights are computed for each batch image
  # 3) loss is computed

  ## l1 classifier
  with tf.variable_scope('l1_classifier'):
    # convert from per dataset class ids to common class ids for l1 classifier in [0, 53]

    # l1 classes organized in group of 10 (0-9), (10-19), ...
    #  0- 9: Bird, Ground Animal, Curb, Fence, Guard Rail, Barrier, Wall, Driveable, Curb Cut, Pedestrian Area,
    # 10-19: Rail Track, Sidewalk, Bridge, Building, Tunnel, Person, Rider, Mountain, Sand, Sky,
    # 20-29: Snow, Terrain, Vegetation, Water, Banner, Bench, Bike Rack, Billboard, CCTV Camera, Fire Hydrant,
    # 30-39: Junction Box, Mailbox, Phone Booth, Street Light, Pole, Traffic Sign Frame, Utility Pole, Traffic Light, Traffic Sign, Trash Can,
    # 40-49: Bicycle, Boat, Bus, Car, Caravan, Motorcycle, On Rails, Other Vehicle, Trailer, Truck,
    # 50-53: Wheeled Slow, Car Mount, Ego Vehicle, Unlabeled

    # 7: driveable, 16: rider, 38: traffic sign
    cids_cityscapes2l1_cids = [52, 7, 11, 7, 10, 13, 6, 3, 4, 12, 14, 34, 37, 38, 22, 21, 19, 15, 16, 43, 49, 42, 44, 48, 46, 45, 40, 53]
    cids_mapillary2l1_cids = [0, 1, 2, 3, 4, 5, 6, 7, 7, 8, 7, 9, 10, 7, 7, 11, 12, 13, 14, 15, 16, 16, 16, 7, 7, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 7, 28, 29, 30, 31, 7, 32, 7, 33, 34, 35, 36, 37, 38, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]
    cids_gtsdb2l1_cids = [38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 53]
    l1_labels_citys = tf.gather(tf.to_int32(cids_cityscapes2l1_cids), labels[0:1])
    l1_labels_mapil = tf.gather(tf.to_int32(cids_mapillary2l1_cids), labels[1:3])
    l1_labels_gtsdb = tf.gather(tf.to_int32(cids_gtsdb2l1_cids), labels[3:4])
    l1_labels = tf.concat([l1_labels_citys, l1_labels_mapil, l1_labels_gtsdb], 0)

    weights = tf.cast(l1_labels <= 52, tf.float32)
    # for l1 classifier, weights for non-per-pixel annotated dataset should be zero
    weights = tf.concat([weights[:3], tf.zeros_like(weights[3:4])], 0)
    weights = weights * lamda[0][0]
    tf.summary.image('l1_weights', weights[..., tf.newaxis]/tf.reduce_max(weights), max_outputs=4, family='debug')
    l1_seg_loss = [_xentr_loss(l1_labels,
                               l1_logits[0],
                               weights,
                               loss_collection=tf.GraphKeys.LOSSES)]

  ## l2 classifiers
  with tf.variable_scope('l2_classifiers'):

    ## l2 driveable classifier
    with tf.variable_scope('l2_driveable_classifier'):
      # bike lane, crosswalk - plain, parking, road, service lane, lane marking - crosswalk, lane marking - general, catch basin, manhole, pothole
      # cityscapes and mapillary has drivable so in order to save computations
      # we deviate from a general implementation to only support those two datasets
      # IMPORTANT: since Cityscapes has only one subclass for the driveable class (parking) nothing pushes the classifier
      # to learn what is not a parking, thus outputing everything as a parking gives the least loss, which is not desirable
      # for now just don't learn from Cityscapes cause it has only one class
      cids_cityscapes2l2_cids_driveable = [10, 10, 10, 2, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10] # cityscapes road --> 10 now
      cids_mapillary2l2_cids_driveable = [10, 10, 10, 10, 10, 10, 10, 0, 1, 10, 2, 10, 10, 3, 4, 10, 10, 10, 10, 10, 10, 10, 10, 5, 6, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 7, 10, 10, 10, 10, 8, 10, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

      l2_labels_citys_driveable = tf.gather(tf.to_int32(cids_cityscapes2l2_cids_driveable), labels[:1])
      l2_labels_mapil_driveable = tf.gather(tf.to_int32(cids_mapillary2l2_cids_driveable), labels[1:3])
      l2_labels_driveable = tf.concat([l2_labels_citys_driveable, l2_labels_mapil_driveable], 0)

      weights = tf.concat([l2_labels_citys_driveable <= 9, l2_labels_mapil_driveable <= 9], 0)
      weights = tf.cast(weights, tf.float32) * lamda[1][0]
      l2_seg_loss_driveable = _xentr_loss(l2_labels_driveable[1:],
                                          l2_logits[0][1:3],
                                          weights[1:],
                                          loss_collection=tf.GraphKeys.LOSSES)

    ## loss for l2 rider classifier
    with tf.variable_scope('l2_rider_classifier'):
      # only mapillary has rider subclasses so in order to save computations
      # we deviate from a general implementation to only support those two datasets
      cids_mapillary2l2_cids_rider = [ 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

      l2_labels_mapil_rider = tf.gather(tf.to_int32(cids_mapillary2l2_cids_rider), labels[1:3])

      weights = tf.cast(l2_labels_mapil_rider <= 2, tf.float32) * lamda[1][1]
      l2_seg_loss_rider = _xentr_loss(l2_labels_mapil_rider,
                                      l2_logits[1][1:3],
                                      weights,
                                      loss_collection=tf.GraphKeys.LOSSES)

    ## loss for l2 traffic sign classifier
    with tf.variable_scope('l2_traffic_sign_classifier'):
      cids_cityscapes2l2_cids_traffic_sign = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2]
      cids_mapillary2l2_cids_traffic_sign = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
      cids_gtsdb2l2_cids_traffic_sign = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]

      l2_labels_citys_traffic_sign = tf.gather(tf.to_int32(cids_cityscapes2l2_cids_traffic_sign), labels[0:1])
      l2_labels_mapil_traffic_sign = tf.gather(tf.to_int32(cids_mapillary2l2_cids_traffic_sign), labels[1:3])
      l2_labels_gtsdb_traffic_sign = tf.gather(tf.to_int32(cids_gtsdb2l2_cids_traffic_sign), labels[3:4])
      l2_labels_traffic_sign = tf.concat([l2_labels_citys_traffic_sign,
                                          l2_labels_mapil_traffic_sign], 0)

      weights = tf.cast(l2_labels_traffic_sign <= 1, tf.float32) * lamda[1][2]
      l2_seg_loss_traffic_sign = _xentr_loss(l2_labels_traffic_sign,
                                             l2_logits[2][0:3],
                                             weights,
                                             loss_collection=tf.GraphKeys.LOSSES)

    l2_seg_loss = [l2_seg_loss_driveable, l2_seg_loss_rider, l2_seg_loss_traffic_sign]

  ## l3 classifier
  with tf.variable_scope('l3_classifier'):
    # only GTSDB here
    # corresponds to bounding box labels thus masking should be done
    with tf.control_dependencies([l2_decs[2]]):
      tp_mask = tf.logical_and(tf.equal(l2_decs[2][3:4], 1),
                               tf.equal(l2_labels_gtsdb_traffic_sign, 1))

    weights = tf.cast(tf.logical_and(tp_mask, labels[3:4] <= 42), tf.float32) * lamda[2][0]

    l3_seg_loss_traffic_sign_front = _xentr_loss(labels[3:4],
                                                 l3_logits[0][3:4],
                                                 weights,
                                                 loss_collection=tf.GraphKeys.LOSSES)
    
    l3_seg_loss = [l3_seg_loss_traffic_sign_front]

  # l2, l3 classifiers will start learning from 2nd epoch
  # gs = tf.train.get_global_step()
  # assert gs, 'Global step not found.'
  # l2_seg_loss_rider = [tf.where(gs<=params.num_batches_per_epoch,
  #                               0.0,
  #                               l/params.loss_regularization[0]) for l in [l2_seg_loss_rider]]
  # l2_seg_loss_traffic_sign = [tf.where(gs<=params.num_batches_per_epoch,
  #                               0.0,
  #                               l/params.loss_regularization[1]) for l in [l2_seg_loss_traffic_sign]]
  # l3_seg_loss_traffic_sign_front = [tf.where(gs<=params.num_batches_per_epoch,
  #                                            0.0,
  #                                            l/params.loss_regularization[2]) for l in [l3_seg_loss_traffic_sign_front]]

  # for l in l2_seg_loss_rider + l2_seg_loss_traffic_sign + l3_seg_loss_traffic_sign_front:
  #   tf.losses.add_loss(l)

  reg_loss = tf.add_n(tf.losses.get_regularization_losses())
  tot_loss = tf.losses.get_total_loss(add_regularization_losses=True)#False)#

  # reg_losses_with_names = []
  # for t in tf.losses.get_regularization_losses():
  #   reg_losses_with_names.append(t.op.name)
  #   reg_losses_with_names.append(t)

  # tot_loss = tf.Print(tot_loss, [l1_seg_loss] + l2_seg_loss + [reg_loss, tot_loss] + reg_losses_with_names, message='losses: ')
  # tot_loss = tf.Print(tot_loss, [l1_seg_loss] + l2_seg_loss + [reg_loss, tot_loss], message='losses: l1, l2, reg, tot: ')
  losses = {'total': tot_loss,
            'segmentation': [l1_seg_loss, l2_seg_loss, l3_seg_loss],
            'regularization': reg_loss}

  return losses
