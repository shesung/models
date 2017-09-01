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

"""Rotated box coder.

"""

import tensorflow as tf
import math

from object_detection.core import box_list

EPSILON = 1e-8


class RotatedBoxCoder(object):

  @property
  def code_size(self):
    return 4

  def encode(self, boxes, rotations, anchors):
    with tf.name_scope('Encode'):
      return self._encode(boxes, rotations, anchors)

  def decode(self, rel_codes, rotations, anchors):
    with tf.name_scope('Decode'):
      return self._decode(rel_codes, rotations, anchors)

  def _encode(self, boxes, rotations, anchors):
    """Encodes a box collection with respect to an anchor collection.

    Args:
      boxes: BoxList holding N boxes to be encoded.
      rotations:
      anchors: BoxList of anchors.

    Returns:
      a tensor representing N anchor-encoded boxes of the format
      [h_top, h_donw, h_left, h_right, theta].
    """
    # Convert anchors to the center coordinate representation.
    ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()
    ycenter, xcenter, h, w = boxes.get_center_coordinates_and_sizes()
    direct_x = xcenter_a - xcenter
    direct_y = ycenter_a - ycenter

    cos_theta = tf.cos(rotations)
    sin_theta = tf.sin(rotations)
    center_x  = tf.multiply(direct_x, cos_theta) + tf.multiply(direct_y, sin_theta)
    center_y  = tf.multiply(direct_y, cos_theta) - tf.multiply(direct_x, sin_theta)

    top   = -0.5 * h - center_y
    left  = -0.5 * w - center_x
    down  = 0.5 * h - center_y
    right = 0.5 * w - center_x

    x_scale = 1.0 / 1024.0
    y_scale = 1.0 / 1024.0
    top   = top   * y_scale
    left  = left  * x_scale
    down  = down  * y_scale
    right = right * x_scale

    return tf.transpose(tf.stack([top, left, down, right, rotations]))

  def _decode(self, rel_codes, rotations, anchors):
    """Decodes relative codes to boxes.

    Args:
      rel_codes: a tensor representing N anchor-encoded boxes.
      anchors: BoxList of anchors.

    Returns:
      boxes: BoxList holding N bounding boxes.
    """
    ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()
    top, left, down, right = tf.unstack(tf.transpose(rel_codes))
    x_scale = 1024.0
    y_scale = 1024.0
    top   = top   * y_scale * -1.0
    left  = left  * x_scale * -1.0
    down  = down  * y_scale
    right = right * x_scale

    rot_center_x = (right + left) * 0.5
    rot_center_y = (top + down) * 0.5
    sin_theta = tf.sin(rotations)
    cos_theta = tf.cos(rotations)
    center_x = tf.multiply(rot_center_x, cos_theta) - tf.multiply(rot_center_y, sin_theta)
    center_y = tf.multiply(rot_center_y, cos_theta) + tf.multiply(rot_center_x, sin_theta)

    abs_center_x = xcenter_a + center_x
    abs_center_y = ycenter_a + center_y

    w = right - left
    h = down - top
    ymin = abs_center_y - 0.5*h
    xmin = abs_center_x - 0.5*w
    ymax = abs_center_y + 0.5*h
    xmax = abs_center_x + 0.5*w

    return box_list.BoxList(tf.transpose(tf.stack([ymin, xmin, ymax, xmax])))
