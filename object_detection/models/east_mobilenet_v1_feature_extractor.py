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

"""SSDFeatureExtractor for MobilenetV1 features."""

import tensorflow as tf

from object_detection.meta_architectures import ssd_meta_arch
from object_detection.models import feature_map_generators
from nets import mobilenet_v1

slim = tf.contrib.slim


class EASTMobileNetV1FeatureExtractor(ssd_meta_arch.SSDFeatureExtractor):
  """SSD Feature Extractor using MobilenetV1 features."""

  def __init__(self,
               depth_multiplier,
               min_depth,
               conv_hyperparams,
               reuse_weights=None):
    """MobileNetV1 Feature Extractor for SSD Models.

    Args:
      depth_multiplier: float depth multiplier for feature extractor.
      min_depth: minimum feature extractor depth.
      conv_hyperparams: tf slim arg_scope for conv2d and separable_conv2d ops.
      reuse_weights: Whether to reuse variables. Default is None.
    """
    super(EASTMobileNetV1FeatureExtractor, self).__init__(
        depth_multiplier, min_depth, conv_hyperparams, reuse_weights)

  def preprocess(self, resized_inputs):
    """SSD preprocessing.

    Maps pixel values to the range [-1, 1].

    Args:
      resized_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    """
    return (2.0 / 255.0) * resized_inputs - 1.0

  def extract_features(self, preprocessed_inputs):
    """Extract features from preprocessed inputs.

    Args:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      feature_maps: a list of tensors where the ith tensor has shape
        [batch, height_i, width_i, depth_i]
    """
    preprocessed_inputs.get_shape().assert_has_rank(4)
    shape_assert = tf.Assert(
        tf.logical_and(tf.greater_equal(tf.shape(preprocessed_inputs)[1], 33),
                       tf.greater_equal(tf.shape(preprocessed_inputs)[2], 33)),
        ['image size must at least be 33 in both height and width.'])

    feature_map_layout = {
        'from_layer': ['east_conv1_3x3'],
        'layer_depth': [-1],
    }

    with tf.control_dependencies([shape_assert]):
      with slim.arg_scope(self._conv_hyperparams):
        with tf.variable_scope('MobilenetV1',
                               reuse=self._reuse_weights) as scope:
          _, image_features = mobilenet_v1.mobilenet_v1_base(
              preprocessed_inputs,
              final_endpoint='Conv2d_13_pointwise',
              min_depth=self._min_depth,
              depth_multiplier=self._depth_multiplier,
              scope=scope)
          """
          by chenx
          """
          east_conv_1 = image_features['Conv2d_3_pointwise']
          east_conv_2 = image_features['Conv2d_5_pointwise']
          east_conv_3 = image_features['Conv2d_11_pointwise']
          east_conv_4 = image_features['Conv2d_13_pointwise']

          east_deconv4 = slim.conv2d_transpose(east_conv_4, 512, [4, 4], 2, \
                                          padding='SAME', scope='east_deconv4')
          east_conv4_concat = tf.concat([east_conv_4, east_deconv4], axis=3)
          east_conv4_1x1 = slim.conv2d(east_conv4_concat, 256, [1,1],
                                       stride=1,
                                       normalizer_fn=slim.batch_norm,
                                       scope='east_conv4_1x1')
          east_conv4_3x3 = slim.conv2d(east_conv4_1x1, 256, [3,3],
                                       stride=1,
                                       normalizer_fn=slim.batch_norm,
                                       scope='east_conv4_3x3')
          image_features['east_conv4_3x3'] = east_conv4_3x3

          east_deconv3 = slim.conv2d_transpose(east_conv4_3x3, 256, [4, 4], 2, \
                                          padding='SAME', scope='east_deconv3')
          east_conv3_concat = tf.concat([east_conv_3, east_deconv3], axis=3)
          east_conv3_1x1 = slim.conv2d(east_conv4_concat, 128, [1,1],
                                       stride=1,
                                       normalizer_fn=slim.batch_norm,
                                       scope='east_conv3_1x1')
          east_conv3_3x3 = slim.conv2d(east_conv4_1x1, 128, [3,3],
                                       stride=1,
                                       normalizer_fn=slim.batch_norm,
                                       scope='east_conv3_3x3')
          image_features['east_conv3_3x3'] = east_conv3_3x3

          east_deconv2 = slim.conv2d_transpose(east_conv3_3x3, 128, [4, 4], 2, \
                                          padding='SAME', scope='east_deconv2')
          east_conv2_concat = tf.concat([east_conv_2, east_deconv3], axis=3)
          east_conv2_1x1 = slim.conv2d(east_conv2_concat, 64, [1,1],
                                       stride=1,
                                       normalizer_fn=slim.batch_norm,
                                       scope='east_conv2_1x1')
          east_conv2_3x3 = slim.conv2d(east_conv2_1x1, 64, [3,3],
                                       stride=1,
                                       normalizer_fn=slim.batch_norm,
                                       scope='east_conv2_3x3')
          image_features['east_conv2_3x3'] = east_conv2_3x3

          east_deconv1 = slim.conv2d_transpose(east_conv2_3x3, 64, [4, 4], 2, \
                                          padding='SAME', scope='east_deconv1')
          east_conv1_concat = tf.concat([east_conv_1, east_deconv1], axis=3)
          east_conv1_1x1 = slim.conv2d(east_conv1_concat, 32, [1,1],
                                       stride=1,
                                       normalizer_fn=slim.batch_norm,
                                       scope='east_conv1_1x1')
          east_conv1_3x3 = slim.conv2d(east_conv1_1x1, 32, [3,3],
                                       stride=1,
                                       normalizer_fn=slim.batch_norm,
                                       scope='east_conv1_3x3')
          image_features['east_conv1_3x3'] = east_conv1_3x3

          feature_maps = feature_map_generators.multi_resolution_feature_maps(
              feature_map_layout=feature_map_layout,
              depth_multiplier=self._depth_multiplier,
              min_depth=self._min_depth,
              insert_1x1_conv=True,
              image_features=image_features)

    return feature_maps.values()
