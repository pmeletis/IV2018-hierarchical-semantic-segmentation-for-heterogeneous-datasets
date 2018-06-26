import tensorflow as tf
# in TF 1.1 metrics_impl has _streaming_confusion_matrix hidden method
from tensorflow.python.ops import metrics_impl

def confusion_matrices_for_classes_and_subclasses(labels, probabilities):
  """VALID ONLY for 19 Cityscapes classes and 43 GTSDB subclasses assuming
  63 trainIds: classIds: 0-18, signIds: 19-61, void: 62 (trafficSignId: 7).
  Calculates the confusion matrix with special care in the traffic sign classes
  and subclasses according to the scheme:
    trainIds 0-18 (classes): all normal except 7 (traffic sign), if a pixel is TP
      top1 labels (7 or 19-61) then it is counted as correct for class 7
    trainIds 19-61 (subclasses): TP if top1 is in 19-61 or top1 is 7 and top2 in 19-61
  
  labels: Nb x H x W, tf.int32, in [0,62]
  probabilities: Nb x H x W x 62, tf.float32, in [0,1]
  """
  probs = probabilities
  # probs.get_shape().assert_is_compatible_with(labels[...,tf.newaxis].get_shape())
  labels.get_shape().assert_has_rank(3)
  probs.get_shape().assert_has_rank(4)
  assert labels.dtype==tf.int32, f"labels dtype is {labels.dtype}"
  assert probs.dtype==tf.float32, f"probs dtype is {probs.dtype}"
  # if labels.dtype != probs.dtype:
  #   assert False, f"labels dtype ({labels.dtype}) doesn't match decisions ({decisions.dtype})"
    #decisions = math_ops.cast(decisions, labels.dtype)
  
  decs = tf.cast(tf.argmax(probs, 3), tf.int32, name='decisions')
  
  # translate ground truth and decisions to classes
  void_mask = tf.equal(labels, 62)
  subclass_mask = tf.logical_and(labels>=19, labels<=61)
  labels2classes = tf.where(void_mask,
                            tf.ones_like(labels)*19,
                            tf.where(subclass_mask,
                                     tf.ones_like(labels)*7,
                                     labels))
  void_mask = tf.equal(decs, 62)
  subclass_mask = tf.logical_and(decs>=19, decs<=61)
  decs2classes = tf.where(void_mask,
                          tf.ones_like(labels)*19,
                          tf.where(subclass_mask,
                                   tf.ones_like(decs)*7,
                                   decs))
  # if label is tsign (7, 19-61) and decision is tsign (7, 19-61) then correct
  # for the rest classes keep as is
  class_cm = metrics_impl._streaming_confusion_matrix(labels2classes,
                                                      decs2classes,
                                                      20)
  
  # translate ground truth and decisions to subclasses
  subclass_mask = tf.logical_and(labels>=19, labels<=61)
  labels2subclasses = tf.where(subclass_mask,
                               labels,
                               tf.ones_like(labels)*62) - 19
  subclass_mask = tf.logical_and(decs>=19, decs<=61)
  tsign_class_mask = tf.equal(decs, 7)
  _, i = tf.nn.top_k(probs, k=2) # 4D
  # print('debug:i:', i.shape, i.dtype)
  top2_is_subclass_mask = tf.logical_and(i[...,1]>=19, i[...,1]<=61) # 3D
  # print('debug:top2_and_any_subclass_mask:', top2_is_subclass_mask.shape, top2_is_subclass_mask.dtype)
  # if top1 is subclass:
  #   keep top1 label
  # else:
  #   if top1 is tsign class:
  #     if top2 is subclass:
  #       keep top2 label
  #     else:
  #       keep top1 label
  #   else:
  #     give void label (because we create labels for subclass evaluation only)
  decs2subclasses = tf.where(subclass_mask,
                             decs,
                             tf.where(tsign_class_mask,
                                      tf.where(top2_is_subclass_mask,
                                               i[...,1],
                                               tf.ones_like(decs)*(7+19)),
                                      tf.ones_like(decs)*62)) - 19
  subclass_cm = metrics_impl._streaming_confusion_matrix(labels2subclasses,
                                                         decs2subclasses,
                                                         44)

  return class_cm, subclass_cm


def mean_iou(labels, decisions, num_classes, params):
  flatten = lambda tensor: tf.reshape(tensor, [-1])

  conf_matrix = tf.confusion_matrix(labels=flatten(labels), predictions=flatten(decisions), num_classes=num_classes)
  if -1 in params.training_problem_def_mapil['lids2cids']:
    conf_matrix = conf_matrix[:-1, :-1]
  inter = tf.diag_part(conf_matrix)
  union = tf.reduce_sum(conf_matrix, 0) + tf.reduce_sum(conf_matrix, 1) - inter

  inter = tf.cast(inter, tf.float32)
  union = tf.cast(union, tf.float32) + 1E-9
  m_iou = tf.reduce_mean(tf.div(inter, union))
  return m_iou