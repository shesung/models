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

"""Tests for object_detection.models.faster_rcnn_resnet_v1_feature_extractor."""

import numpy as np
import tensorflow as tf

from object_detection.models import east_mobilenet_v1_feature_extractor


class EASTMobilenetV1FeatureExtractorTest(tf.test.TestCase):

  def setUp(self):
    self.feature_extractor = east_mobilenet_v1_feature_extractor.EASTMobileNetV1FeatureExtractor(
        min_depth = 8,
        depth_multiplier = 1.0,
        conv_hyperparams={})

  def test_extract_proposal_features_returns_expected_size(self):
    preprocessed_inputs = tf.random_uniform(
        [4, 256, 256, 3], maxval=255, dtype=tf.float32)
    feature_map = self.feature_extractor.extract_features(preprocessed_inputs)[0]
    features_shape = tf.shape(feature_map)

    init_op = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init_op)
      features_shape_out = sess.run(features_shape)
      self.assertAllEqual(features_shape_out, [4, 64, 64, 32])

  def test_extract_proposal_features_half_size_input(self):
    preprocessed_inputs = tf.random_uniform(
        [1, 128, 128, 3], maxval=255, dtype=tf.float32)
    feature_map = self.feature_extractor.extract_features(preprocessed_inputs)[0]
    features_shape = tf.shape(feature_map)

    init_op = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init_op)
      features_shape_out = sess.run(features_shape)
      self.assertAllEqual(features_shape_out, [1, 32, 32, 32])


  def test_extract_proposal_features_dies_on_very_small_images(self):
    preprocessed_inputs = tf.placeholder(tf.float32, (4, None, None, 3))
    feature_map = self.feature_extractor.extract_features(preprocessed_inputs)[0]
    features_shape = tf.shape(feature_map)

    init_op = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init_op)
      with self.assertRaises(tf.errors.InvalidArgumentError):
        sess.run(
            features_shape,
            feed_dict={preprocessed_inputs: np.random.rand(4, 32, 32, 3)})

if __name__ == '__main__':
  tf.test.main()
