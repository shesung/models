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

"""Tests for ssd_mobilenet_v1_feature_extractor."""
import numpy as np
import sys
sys.path.insert(0, "/home/chenxiang/code/tensorflow/_python_build")

import tensorflow as tf

from object_detection.models import ssd_feature_extractor_test
from object_detection.models import east_mobilenet_v1_feature_extractor


class EASTMobilenetV1FeatureExtractorTest(
    ssd_feature_extractor_test.SsdFeatureExtractorTestBase, tf.test.TestCase):

  def _create_feature_extractor(self, depth_multiplier):
    """Constructs a new feature extractor.

    Args:
      depth_multiplier: float depth multiplier for feature extractor
    Returns:
      an ssd_meta_arch.SSDFeatureExtractor object.
    """
    min_depth = 32
    conv_hyperparams = {}
    return east_mobilenet_v1_feature_extractor.EASTMobileNetV1FeatureExtractor(
        depth_multiplier, min_depth, conv_hyperparams)

  def test_extract_features_returns_correct_shapes_128(self):
    image_height = 128
    image_width = 128
    depth_multiplier = 1.0
    expected_feature_map_shape = [(4, 32, 32, 32)]
    self.check_extract_features_returns_correct_shape(
        image_height, image_width, depth_multiplier, expected_feature_map_shape)

  def test_extract_features_returns_correct_shapes_299(self):
    image_height = 512
    image_width = 512
    depth_multiplier = 1.0
    expected_feature_map_shape = [(4, 128, 128, 32)]
    self.check_extract_features_returns_correct_shape(
        image_height, image_width, depth_multiplier, expected_feature_map_shape)

  def test_extract_features_raises_error_with_invalid_image_size(self):
    image_height = 32
    image_width = 32
    depth_multiplier = 1.0
    self.check_extract_features_raises_error_with_invalid_image_size(
        image_height, image_width, depth_multiplier)

  def test_preprocess_returns_correct_value_range(self):
    image_height = 128
    image_width = 128
    depth_multiplier = 1
    test_image = np.random.rand(4, image_height, image_width, 3)
    feature_extractor = self._create_feature_extractor(depth_multiplier)
    preprocessed_image = feature_extractor.preprocess(test_image)
    self.assertTrue(np.all(np.less_equal(np.abs(preprocessed_image), 1.0)))

  def test_variables_only_created_in_scope(self):
    depth_multiplier = 1
    scope_name = 'MobilenetV1'
    self.check_feature_extractor_variables_under_scope(depth_multiplier,
                                                       scope_name)


if __name__ == '__main__':
  tf.test.main()
