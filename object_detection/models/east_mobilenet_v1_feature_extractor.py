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

"""EASTFeatureExtractor for MobilenetV1 features."""

import tensorflow as tf

from object_detection.meta_architectures import east_meta_arch
from object_detection.models import feature_map_generators
from slim.nets import mobilenet_v1

slim = tf.contrib.slim


class EASTMobileNetV1FeatureExtractor(east_meta_arch.EASTFeatureExtractor):
  """EAST Feature Extractor using MobilenetV1 features."""

  def __init__(self,
               depth_multiplier,
               min_depth,
               conv_hyperparams,
               reuse_weights=None):
    """MobileNetV1 Feature Extractor for EAST Models.

    Args:
      depth_multiplier: float depth multiplier for feature extractor.
      min_depth: minimum feature extractor depth.
      conv_hyperparams: tf slim arg_scope for conv2d and separable_conv2d ops.
      reuse_weights: Whether to reuse variables. Default is None.
    """
    super(EASTMobileNetV1FeatureExtractor, self).__init__(
        depth_multiplier, min_depth, conv_hyperparams, reuse_weights)

  def preprocess(self, resized_inputs):
    """EAST preprocessing.

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

    with tf.control_dependencies([shape_assert]):
      with slim.arg_scope(self._conv_hyperparams):
        with tf.variable_scope('MobilenetV1',
                               reuse=self._reuse_weights) as scope:
          preprocessed_inputs = tf.Print(preprocessed_inputs, [tf.shape(preprocessed_inputs)], message='preprocessed_inputs=')
          _, image_features = mobilenet_v1.mobilenet_v1_base(
              preprocessed_inputs,
              final_endpoint='Conv2d_13_pointwise',
              min_depth=self._min_depth,
              depth_multiplier=self._depth_multiplier,
              scope=scope)
          """
          by chenx
          """
          feature_map = self.merge_from_multi_feature_maps(image_features)
    return [feature_map]

  def merge_from_multi_feature_maps(self, feature_maps):
    east_conv_1 = feature_maps['Conv2d_3_pointwise']
    east_conv_2 = feature_maps['Conv2d_5_pointwise']
    east_conv_3 = feature_maps['Conv2d_9_pointwise']
    east_conv_4 = feature_maps['Conv2d_13_pointwise']
    east_conv_1 = tf.Print(east_conv_1, [tf.shape(east_conv_1)], message='east_conv_1=')
    east_conv_2 = tf.Print(east_conv_2, [tf.shape(east_conv_2)], message='east_conv_2=')
    east_conv_3 = tf.Print(east_conv_3, [tf.shape(east_conv_3)], message='east_conv_3=')
    east_conv_4 = tf.Print(east_conv_4, [tf.shape(east_conv_4)], message='east_conv_4=')

    with tf.variable_scope("MergeBranch"):
      east_deconv4 = slim.conv2d_transpose(east_conv_4, 512, [4, 4], 2,
                                           padding='SAME', scope='east_deconv4')
      east_deconv4 = tf.Print(east_deconv4, [tf.shape(east_deconv4)], message='east_deconv4=')
      east_conv4_concat = tf.concat([east_conv_3, east_deconv4], axis=3)
      east_conv4_1x1 = slim.conv2d(east_conv4_concat, 128, [1,1],
                                   stride=1,
                                   normalizer_fn=slim.batch_norm,
                                   scope='east_conv4_1x1')
      east_conv4_3x3 = slim.conv2d(east_conv4_1x1, 128, [3,3],
                                   stride=1,
                                   normalizer_fn=slim.batch_norm,
                                   scope='east_conv4_3x3')

      east_deconv3 = slim.conv2d_transpose(east_conv4_3x3, 128, [4, 4], 2,
                                      padding='SAME', scope='east_deconv3')
      east_deconv3 = tf.Print(east_deconv3, [tf.shape(east_deconv3)], message='east_deconv3=')
      east_conv3_concat = tf.concat([east_conv_2, east_deconv3], axis=3)
      east_conv3_1x1 = slim.conv2d(east_conv3_concat, 64, [1,1],
                                   stride=1,
                                   normalizer_fn=slim.batch_norm,
                                   scope='east_conv3_1x1')
      east_conv3_3x3 = slim.conv2d(east_conv3_1x1, 64, [3,3],
                                   stride=1,
                                   normalizer_fn=slim.batch_norm,
                                   scope='east_conv3_3x3')

      east_deconv2 = slim.conv2d_transpose(east_conv3_3x3, 64, [4, 4], 2, \
                                      padding='SAME', scope='east_deconv2')
      east_deconv2 = tf.Print(east_deconv2, [tf.shape(east_deconv2)], message='east_deconv2=')
      east_conv2_concat = tf.concat([east_conv_1, east_deconv2], axis=3)
      east_conv2_1x1 = slim.conv2d(east_conv2_concat, 32, [1,1],
                                   stride=1,
                                   normalizer_fn=slim.batch_norm,
                                   scope='east_conv2_1x1')
      east_conv2_3x3 = slim.conv2d(east_conv2_1x1, 32, [3,3],
                                   stride=1,
                                   normalizer_fn=slim.batch_norm,
                                   scope='east_conv2_3x3')
      east_conv2_3x3 = tf.Print(east_conv2_3x3, [east_conv2_3x3, tf.shape(east_conv2_3x3)],
              message='east_conv2_3x3=')
      east_conv1_3x3 = slim.conv2d(east_conv2_3x3, 32, [3,3],
                                   stride=1,
                                   normalizer_fn=slim.batch_norm,
                                   scope='east_conv1_3x3')
    return east_conv1_3x3
