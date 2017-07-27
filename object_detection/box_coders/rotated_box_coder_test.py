"""Tests for object_detection.box_coder.rotated_box_coder."""
import sys
sys.path.insert(0, "/home/chenxiang/code/tensorflow/_python_build")

import tensorflow as tf

from object_detection.box_coders import rotated_box_coder
from object_detection.core import box_list
import math

class RotatedBoxCoderTest(tf.test.TestCase):

  def test_encode(self):
    boxes = [[10.0, 10.0, 20.0, 30.0],
             [10.0, 10.0, 20.0, 30.0],
             [10.0, 10.0, 20.0, 30.0]]
    anchors = [[14.0, 19.0, 16.0, 21.0],
               [13.0, 18.0, 15.0, 20.0],
               [13.0, 18.0, 15.0, 20.0]]
    rotations = [0.0, 0.0, math.pi*0.5]
    expected_rel_codes = [[5.0, 5.0, 10.0, 10.0, 0.0],
                          [4.0, 6.0, 9.0, 11.0, 0.0],
                          [6.0, 4.0, 9.0, 11.0, math.pi*0.5]]

    boxes = box_list.BoxList(tf.constant(boxes))
    anchors = box_list.BoxList(tf.constant(anchors))
    rot = tf.constant(rotations)
    coder = rotated_box_coder.RotatedBoxCoder()
    rel_codes = coder.encode(boxes, rotations, anchors)
    with self.test_session() as sess:
      (rel_codes_out,) = sess.run([rel_codes])
      self.assertAllClose(rel_codes_out, expected_rel_codes)

  def test_decode(self):
    boxes = [[20.0, 20.0, 10.0, 10.0],
             [10.0, 30.0, 10.0, 10.0],
             [10.0, 10.0, 20.0, 30.0]]
    anchors = [[50.0, 50.0, 70.0, 90.0],
               [50.0, 70.0, 70.0, 90.0],
               [13.0, 18.0, 15.0, 20.0]]
    rotations = [math.pi*0.5, math.pi*0.5,0]
    expected_rel_codes = [[40.0, 60.0, 80.0, 80.0],
                          [40, 60.0, 80.0, 80.0],
                          [6.0, 4.0, 9.0, 11.0]]

    anchors = box_list.BoxList(tf.constant(anchors))
    coder = rotated_box_coder.RotatedBoxCoder()
    rel_codes = coder.decode(boxes, rotations, anchors)
    with self.test_session() as sess:
      (rel_codes_out,) = sess.run([rel_codes])
      self.assertAllClose(rel_codes_out, expected_rel_codes)

if __name__ == '__main__':
  tf.test.main()
