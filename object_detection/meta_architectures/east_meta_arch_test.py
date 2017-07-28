# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for object_detection.meta_architectures.ssd_meta_arch."""
import functools
import numpy as np
import tensorflow as tf

from tensorflow.python.training import saver as tf_saver
from object_detection.core import anchor_generator
from object_detection.core import box_coder
from object_detection.core import box_predictor
from object_detection.core import box_list
from object_detection.core import losses
from object_detection.core import post_processing
from object_detection.meta_architectures import east_meta_arch

slim = tf.contrib.slim


class FakeEASTFeatureExtractor(east_meta_arch.EASTFeatureExtractor):

  def __init__(self):
    super(FakeEASTFeatureExtractor, self).__init__(
        depth_multiplier=0, min_depth=0, conv_hyperparams=None)

  def preprocess(self, resized_inputs):
    return tf.identity(resized_inputs)

  def extract_features(self, preprocessed_inputs):
    with tf.variable_scope('mock_model'):
      features = slim.conv2d(inputs=preprocessed_inputs, num_outputs=32,
                             kernel_size=[1, 1], scope='layer1')
      return [features]


class MockAnchorGenerator2x2(anchor_generator.AnchorGenerator):
  """Sets up a simple 2x2 anchor grid on the unit square."""

  def name_scope(self):
    return 'MockAnchorGenerator'

  def num_anchors_per_location(self):
    return [1]

  def _generate(self, feature_map_shape_list):
    return box_list.BoxList(
        tf.constant([[0, 0, 4, 4],
                     [0, 4, 4, 8],
                     [4, 0, 8, 4],
                     [4, 4, 8, 8]], tf.float32))

class MockBoxCoder(box_coder.BoxCoder):

  @property
  def code_size(self):
    return 4

  def _encode(self, boxes, rotations, anchors):
    (ycenter, xcenter, height, width) = anchors.get_center_coordinates_and_sizes()
    gt_boxes = boxes.get()
    h1 = gt_boxes[:,0] - ycenter
    h2 = gt_boxes[:,1] - xcenter
    h3 = gt_boxes[:,2] - ycenter
    h4 = gt_boxes[:,3] - xcenter
    return tf.transpose(tf.stack([h1, h2, h3, h4, rotations]))

  def _decode(self, rel_codes, rotations, anchors):
    (ycenter, xcenter, height, width) = anchors.get_center_coordinates_and_sizes()
    ymin = ycenter + rel_codes[:, 0]
    xmin = xcenter + rel_codes[:, 1]
    ymax = ycenter + rel_codes[:, 2]
    xmax = xcenter + rel_codes[:, 3]
    return box_list.BoxList(tf.transpose(tf.stack([ymin, xmin, ymax, xmax])))

  def encode(self, boxes, rotations, anchors):
    with tf.name_scope('Encode'):
      return self._encode(boxes, rotations, anchors)

  def decode(self, rel_codes, rotations, anchors):
    with tf.name_scope('Decode'):
      return self._decode(rel_codes, rotations, anchors)


class MockBoxPredictor(box_predictor.BoxPredictor):
  """Simple box predictor that ignores inputs and outputs all zeros."""

  def __init__(self, is_training, num_classes):
    super(MockBoxPredictor, self).__init__(is_training, num_classes)

  def _predict(self, image_features, num_predictions_per_location):
    batch_size = image_features.get_shape().as_list()[0]
    num_anchors = (image_features.get_shape().as_list()[1]
                   * image_features.get_shape().as_list()[2])
    code_size = 4
    zero = tf.reduce_sum(0 * image_features)
    # box_encodings: [top, down, left, right]
    box_encodings = zero + tf.concat([
        -2 * tf.ones((batch_size, num_anchors, 1, 1), dtype=tf.float32),
        -2 * tf.ones((batch_size, num_anchors, 1, 1), dtype=tf.float32),
        2 * tf.ones((batch_size, num_anchors, 1, 1), dtype=tf.float32),
        2 * tf.ones((batch_size, num_anchors, 1, 1), dtype=tf.float32)],
        axis=-1)
    score_predictions = zero + tf.zeros(
        (batch_size, num_anchors, 1), dtype=tf.float32)
    angle_encodings = zero + tf.zeros(
        (batch_size, num_anchors, 1, 1), dtype=tf.float32)
    return {box_predictor.BOX_ENCODINGS: box_encodings,
            box_predictor.ANGLE_ENCODINGS: angle_encodings,
            box_predictor.SCORE_PREDICTIONS: score_predictions}

class EASTMetaArchTest(tf.test.TestCase):

  def setUp(self):
    """Set up mock SSD model.

    Here we set up a simple mock SSD model that will always predict 4
    detections that happen to always be exactly the anchors that are set up
    in the above MockAnchorGenerator.  Because we let max_detections=5,
    we will also always end up with an extra padded row in the detection
    results.
    """
    is_training = False
    self._num_classes = 1
    mock_anchor_generator = MockAnchorGenerator2x2()
    mock_box_predictor = MockBoxPredictor(is_training, self._num_classes)
    mock_box_coder = MockBoxCoder()
    fake_feature_extractor = FakeEASTFeatureExtractor()

    def image_resizer_fn(image):
      return tf.identity(image)

    classification_loss = losses.ScoreLoss()
    localization_loss = losses.RBoxLocalizationLoss(alpha=1.0)
    non_max_suppression_fn = functools.partial(
        post_processing.batch_multiclass_non_max_suppression,
        score_thresh=0.1,
        iou_thresh=1.0,
        max_size_per_class=5,
        max_total_size=5)
    classification_loss_weight = 1.0
    localization_loss_weight = 1.0
    normalize_loss_by_num_matches = False


    self._num_anchors = 4
    self._code_size = 4
    self._model = east_meta_arch.EASTMetaArch(
        is_training,
        mock_anchor_generator,
        mock_box_predictor,
        mock_box_coder,
        fake_feature_extractor,
        image_resizer_fn,
        non_max_suppression_fn,
        tf.sigmoid,
        classification_loss,
        localization_loss,
        classification_loss_weight,
        localization_loss_weight,
        normalize_loss_by_num_matches)

  def test_predict_results_have_correct_keys_and_shapes(self):
    batch_size = 1
    preprocessed_input = tf.random_uniform((batch_size, 2, 2, 3),
                                           dtype=tf.float32)
    prediction_dict = self._model.predict(preprocessed_input)

    self.assertTrue('box_encodings' in prediction_dict)
    self.assertTrue('rotations' in prediction_dict)
    self.assertTrue('scores' in prediction_dict)
    self.assertTrue('feature_maps' in prediction_dict)

    expected_box_encodings_shape_out = (
        batch_size, self._num_anchors, self._code_size)
    expected_scores_shape_out = (
        batch_size, self._num_anchors, 1)
    expected_rotations_shape_out = (
        batch_size, self._num_anchors, 1)
    init_op = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init_op)
      prediction_out = sess.run(prediction_dict)
      self.assertAllEqual(
        prediction_out['box_encodings'].shape,
        expected_box_encodings_shape_out)
      self.assertAllEqual(
        prediction_out['scores'].shape,
        expected_scores_shape_out)
      self.assertAllEqual(
        prediction_out['rotations'].shape,
        expected_scores_shape_out)

  def test_postprocess_results_are_correct(self):
    batch_size = 2
    preprocessed_input = tf.random_uniform((batch_size, 2, 2, 3),
                                           dtype=tf.float32)
    prediction_dict = self._model.predict(preprocessed_input)
    detections = self._model.postprocess(prediction_dict)

    expected_boxes = np.array([[[0, 0, 4, 4],
                                [0, 4, 4, 8],
                                [4, 0, 8, 4],
                                [4, 4, 8, 8],
                                [0, 0, 0, 0]],
                               [[0, 0, 4, 4],
                                [0, 4, 4, 8],
                                [4, 0, 8, 4],
                                [4, 4, 8, 8],
                                [0, 0, 0, 0]]], dtype=np.float32)
    expected_scores = 0.5 * np.array([[1, 1, 1, 1, 0],
                                      [1, 1, 1, 1, 0]])
    expected_classes = np.array([[0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0]])
    expected_num_detections = np.array([4, 4])

    self.assertTrue('detection_boxes' in detections)
    self.assertTrue('detection_scores' in detections)
    self.assertTrue('detection_classes' in detections)
    self.assertTrue('num_detections' in detections)

    init_op = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init_op)
      detections_out = sess.run(detections)
      self.assertAllClose(detections_out['detection_boxes'], expected_boxes)
      self.assertAllClose(detections_out['detection_scores'], expected_scores)
      self.assertAllClose(detections_out['detection_classes'], expected_classes)
      self.assertAllClose(detections_out['num_detections'],
                          expected_num_detections)

  def test_loss_results_are_correct(self):
    batch_size = 2
    preprocessed_input = tf.random_uniform((batch_size, 2, 2, 3),
                                           dtype=tf.float32)
    groundtruth_boxes_list = [tf.constant([[0, 0, 4, 4]], dtype=tf.float32),
                              tf.constant([[0, 0, 4, 4]], dtype=tf.float32)]
    groundtruth_masks_list = [tf.constant([[[1, 0],
                                            [0, 0]]], dtype=tf.int32),
                              tf.constant([[[1, 0],
                                            [0, 0]]], dtype=tf.int32)]
    groundtruth_rotations_list = [tf.constant([0], dtype=tf.float32),
                                  tf.constant([0], dtype=tf.float32)]
    self._model.provide_groundtruth(groundtruth_boxes_list,
                                    None,
                                    groundtruth_masks_list=groundtruth_masks_list,
                                    groundtruth_rotations_list=groundtruth_rotations_list)
    prediction_dict = self._model.predict(preprocessed_input)
    loss_dict = self._model.loss(prediction_dict)

    self.assertTrue('localization_loss' in loss_dict)
    self.assertTrue('classification_loss' in loss_dict)

    expected_localization_loss = 0.0
    expected_classification_loss = batch_size * (-0.75 * np.log(0.5) -
                                                 0.25 * 3 * np.log(1-0.5))
    init_op = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init_op)
      losses_out = sess.run(loss_dict)

      self.assertAllClose(losses_out['localization_loss'],
                          expected_localization_loss)
      self.assertAllClose(losses_out['classification_loss'],
                          expected_classification_loss)

  def test_restore_fn_detection(self):
    init_op = tf.global_variables_initializer()
    saver = tf_saver.Saver()
    save_path = self.get_temp_dir()
    with self.test_session() as sess:
      sess.run(init_op)
      saved_model_path = saver.save(sess, save_path)
      restore_fn = self._model.restore_fn(saved_model_path,
                                          from_detection_checkpoint=True)
      restore_fn(sess)
      for var in sess.run(tf.report_uninitialized_variables()):
        self.assertNotIn('FeatureExtractor', var.name)

  def test_restore_fn_classification(self):
    # Define mock tensorflow classification graph and save variables.
    test_graph_classification = tf.Graph()
    with test_graph_classification.as_default():
      image = tf.placeholder(dtype=tf.float32, shape=[1, 20, 20, 3])
      with tf.variable_scope('mock_model'):
        net = slim.conv2d(image, num_outputs=32, kernel_size=1, scope='layer1')
        slim.conv2d(net, num_outputs=3, kernel_size=1, scope='layer2')

      init_op = tf.global_variables_initializer()
      saver = tf.train.Saver()
      save_path = self.get_temp_dir()
      with self.test_session() as sess:
        sess.run(init_op)
        saved_model_path = saver.save(sess, save_path)

    # Create tensorflow detection graph and load variables from
    # classification checkpoint.
    test_graph_detection = tf.Graph()
    with test_graph_detection.as_default():
      inputs_shape = [2, 2, 2, 3]
      inputs = tf.to_float(tf.random_uniform(
          inputs_shape, minval=0, maxval=255, dtype=tf.int32))
      preprocessed_inputs = self._model.preprocess(inputs)
      prediction_dict = self._model.predict(preprocessed_inputs)
      self._model.postprocess(prediction_dict)
      restore_fn = self._model.restore_fn(saved_model_path,
                                          from_detection_checkpoint=False)
      with self.test_session() as sess:
        restore_fn(sess)
        for var in sess.run(tf.report_uninitialized_variables()):
          self.assertNotIn('FeatureExtractor', var.name)

if __name__ == '__main__':
  tf.test.main()
