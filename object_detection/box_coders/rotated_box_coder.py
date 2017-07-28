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

    top   = 0.5 * h + center_y
    down  = 0.5 * h - center_y
    left  = 0.5 * w + center_x
    right = 0.5 * w - center_x
    return tf.transpose(tf.stack([top, down, left, right, rotations]))

  def _decode(self, rel_codes, rotations, anchors):
    """Decodes relative codes to boxes.

    Args:
      rel_codes: a tensor representing N anchor-encoded boxes.
      anchors: BoxList of anchors.

    Returns:
      boxes: BoxList holding N bounding boxes.
    """
    ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()
    top, down, left, right = tf.unstack(tf.transpose(rel_codes))

    diff_x = (right - left)*0.5
    diff_y = (top - down)*0.5
    #x1 = diff_x - left
    #y1 = diff_y - down
    #x2 = right + diff_x
    #y2 = top + diff_y
    sin_theta = tf.sin(rotations)
    cos_theta = tf.cos(rotations)
    dx  = tf.multiply(diff_x, cos_theta) + tf.multiply(diff_y, sin_theta)
    dy  = tf.multiply(diff_y, cos_theta) - tf.multiply(diff_x, sin_theta)

    center_x = xcenter_a + dx
    center_y = ycenter_a + dy
    #x1_std  = tf.multiply(x1, tf.cos(rotations)) + tf.multiply(y1, tf.sin(rotations))
    #y1_std  = tf.multiply(y1, tf.cos(rotations)) + tf.multiply(x1, tf.sin(rotations))
    #x2_std  = tf.multiply(x2, tf.cos(rotations)) + tf.multiply(y2, tf.sin(rotations))
    #y2_std  = tf.multiply(y2, tf.cos(rotations)) + tf.multiply(x2, tf.sin(rotations))

    #dx = xcenter_a - diff_x_std
    #dy = ycenter_a - diff_y_std
    w = left + right
    h = top + down
    xmin = center_x - 0.5*w
    ymin = center_y - 0.5*h
    xmax = center_x + 0.5*w
    ymax = center_y + 0.5*h

    return tf.transpose(tf.stack([ymin, xmin, ymax, xmax]))
