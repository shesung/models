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

"""Square box coder.

Square box coder follows the coding schema described below:
l = sqrt(h * w)
la = sqrt(ha * wa)
ty = (y - ya) / la
tx = (x - xa) / la
tl = log(l / la)
where x, y, w, h denote the box's center coordinates, width, and height,
respectively. Similarly, xa, ya, wa, ha denote the anchor's center
coordinates, width and height. tx, ty, tl denote the anchor-encoded
center, and length, respectively. Because the encoded box is a square, only
one length is encoded.

This has shown to provide performance improvements over the Faster RCNN box
coder when the objects being detected tend to be square (e.g. faces) and when
the input images are not distorted via resizing.
"""

import tensorflow as tf

from object_detection.core import box_list

EPSILON = 1e-8


class EastBoxCoder(object):
  """Encodes a 3-scalar representation of a square box."""

  def __init__(self, scale_factors=None):
    """Constructor for SquareBoxCoder.

    Args:
      scale_factors: List of 3 positive scalars to scale ty, tx, and tl.
        If set to None, does not perform scaling. For faster RCNN,
        the open-source implementation recommends using [10.0, 10.0, 5.0].

    Raises:
      ValueError: If scale_factors is not length 3 or contains values less than
        or equal to 0.
    """
    if scale_factors:
      if len(scale_factors) != 3:
        raise ValueError('The argument scale_factors must be a list of length '
                         '3.')
      if any(scalar <= 0 for scalar in scale_factors):
        raise ValueError('The values in scale_factors must all be greater '
                         'than 0.')
    self._scale_factors = scale_factors

  def encode(self, boxes, rotations, anchors):
    """Encodes a box collection with respect to an anchor collection.

    Args:
      boxes: BoxList holding N boxes to be encoded.
      anchors: BoxList of anchors.

    Returns:
      a tensor representing N anchor-encoded boxes of the format
      [ty, tx, tl].
    """
    # Convert anchors to the center coordinate representation.
    ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()
    ycenter, xcenter, h, w = boxes.get_center_coordinates_and_sizes()
	direct_x = xcenter_a - xcenter
	direct_y = ycenter_a - ycenter

	center_x  = tf.multiply(direct_x, tf.cos(rotations)) +
					tf.multiply(direct_y, tf.sin(rotations))
	center_y  = tf.multiply(direct_y, tf.cos(rotations)) +
					tf.multiply(direct_x, tf.sin(rotations))
	# Avoid NaN in division and log below.
    la += EPSILON
    l += EPSILON

    top = tf.abs(0.5*h - center_y)
    bown = tf.abs(0.5*h + center_y)
    left = tf.abs(0.5*w + center_x)
    right = tf.abs(0.5*w - center_x)
    # Scales location targets for joint training.
    if self._scale_factors:
      top *= self._scale_factors[0]
      bown *= self._scale_factors[0]
      left *= self._scale_factors[1]
      right *= self._scale_factors[1]
    return tf.transpose(tf.stack([top, bown, left, right]))

  def _decode(self, rel_codes, rotaions, anchors):
    """Decodes relative codes to boxes.

    Args:
      rel_codes: a tensor representing N anchor-encoded boxes.
      anchors: BoxList of anchors.

    Returns:
      boxes: BoxList holding N bounding boxes.
    """
    ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()
    top, bown, left, right = tf.unstack(tf.transpose(rel_codes))
    if self._scale_factors:
      top /= self._scale_factors[0]
      bown /= self._scale_factors[0]
      left /= self._scale_factors[1]
      right /= self._scale_factors[1]
    ymin = ycenter_a - top
    xmin = xcenter_a - left
    ymax = ycenter_a + bown
    xmax = xcenter_a + right
    return box_list.BoxList(tf.transpose(tf.stack([ymin, xmin, ymax, xmax])))
