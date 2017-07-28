"""Tests for object_detection.box_coder.rotated_box_coder."""

import tensorflow as tf

from object_detection.box_coders import rotated_box_coder
from object_detection.core import box_list
import math

class RotatedBoxCoderTest(tf.test.TestCase):

  def test_encode(self):
    boxes = [[0, 0, 2, 4],
             [0, 0, 2, 4],
             [0, 0, 2, 4]]
    anchors = [[0, 0, 2, 4],
               [0, 0, 1, 2],
               [0, 0, 1, 2]]
    rotations = [0.0, 0.0, math.pi*0.5]
    expected_rel_codes = [[-1, -2, 1, 2, 0.0],
                          [-0.5, -1, 1.5, 3, 0.0],
                          [-2, -1.5, 0, 2.5, math.pi*0.5]]

    boxes = box_list.BoxList(tf.constant(boxes, tf.float32))
    anchors = box_list.BoxList(tf.constant(anchors, tf.float32))
    rot = tf.constant(rotations, tf.float32)
    coder = rotated_box_coder.RotatedBoxCoder()
    rel_codes = coder.encode(boxes, rotations, anchors)
    with self.test_session() as sess:
      (rel_codes_out,) = sess.run([rel_codes])
      self.assertAllClose(rel_codes_out, expected_rel_codes)

  def test_decode(self):
    rel_codes = [[-1, -2, 1, 2],
                 [-0.5, -1, 1.5, 3],
                 [-2, -1.5, 0, 2.5]]
    anchors = [[0, 0, 2, 4],
               [0, 0, 1, 2],
               [0, 0, 1, 2]]
    rotations = [0.0, 0.0, math.pi*0.5]
    expected_boxes = [[0, 0, 2, 4],
                      [0, 0, 2, 4],
                      [0, 0, 2, 4]]
    anchors = box_list.BoxList(tf.constant(anchors, tf.float32))
    rotations = tf.constant(rotations, tf.float32)
    coder = rotated_box_coder.RotatedBoxCoder()
    boxes = coder.decode(rel_codes, rotations, anchors)
    with self.test_session() as sess:
      (boxes_out,) = sess.run([boxes.get()])
      self.assertAllClose(boxes_out, expected_boxes)

if __name__ == '__main__':
  tf.test.main()
