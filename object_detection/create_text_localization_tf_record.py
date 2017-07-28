#!/usr/bin/env python

r"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    python create_text_localization_tf_record.py \
        --output_path=/home/user/pascal.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os

import numpy as np
from skimage.draw import polygon

from lxml import etree
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


flags = tf.app.flags

flags.DEFINE_string('data_dir', '/world/data-c7/censhusheng/data/MSRA-TD500/train', 'Root directory to images')
flags.DEFINE_string('output_path', './text_loc.record', 'Path to output TFRecord')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                    'merged set.')

FLAGS = flags.FLAGS
SETS = ['train', 'val', 'trainval', 'test']


def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    dataset_directory: Path to root directory holding PASCAL dataset
    label_map_dict: A map from string label names to integers ids.
    image_subdirectory: String specifying subdirectory within the
      PASCAL dataset directory holding the actual image data.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  #img_path = os.path.join(data['folder'], image_subdirectory, data['filename'])
  full_path = os.path.join(dataset_directory, data['filename'])
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  width, height = image.size

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  rotation = []
  mask = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  for obj in data['object']:
    difficult = bool(int(obj['difficult']))
    difficult_obj.append(int(difficult))

    # we use absolute coordinates in east pipeline
    xmin.append(float(obj['bndbox']['xmin']))
    ymin.append(float(obj['bndbox']['ymin']))
    xmax.append(float(obj['bndbox']['xmax']))
    ymax.append(float(obj['bndbox']['ymax']))
    rotation.append(float(obj['rotation']))
    mask = mask + obj['mask']
    classes_text.append(obj['name'].encode('utf8'))
    classes.append(label_map_dict[obj['name']])
    truncated.append(int(obj['truncated']))
    poses.append(obj['pose'].encode('utf8'))

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/rotation': dataset_util.float_list_feature(rotation),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
      'image/segmentation/object': dataset_util.int64_list_feature(mask),
      'image/segmentation/object/class': dataset_util.int64_list_feature(classes),
  }))
  return example


def rbox_2_polygon(x, y, w, h, rad):
    w2 = w*0.5
    h2 = h*0.5
    xc = x + w2
    yc = y + h2
    m_rot = np.array([[np.cos(rad), -np.sin(rad)],
                      [np.sin(rad), np.cos(rad)]], dtype=np.float32)
    pts_ = np.array([[-w2,-h2],
                     [w2,-h2],
                     [w2,h2],
                     [-w2,h2]], dtype=np.float32)
    pts = np.dot(m_rot, pts_.T).T
    pts[:,0] = pts[:,0] + xc
    pts[:,1] = pts[:,1] + yc
    return pts.flatten().tolist()


def main(_):
    if FLAGS.set not in SETS:
        raise ValueError('set must be in : {}'.format(SETS))

    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    label_map_dict = label_map_util.get_label_map_dict('object_detection/data/text_label_map.pbtxt')

    data_dir = FLAGS.data_dir
    for fn in os.listdir(data_dir)[:16]:
        if os.path.splitext(fn)[1] == '.gt':
            continue

        im = PIL.Image.open(os.path.join(data_dir, fn))
        im_width, im_height = im.size
        gt_fn = os.path.splitext(fn)[0] + '.gt'
        obj_list = []
        with open(os.path.join(data_dir, gt_fn)) as f_gt:
            for line in f_gt.readlines():
                #line = line.decode("utf-8").encode("utf-8")
                items = line.strip().split()
                index = int(items[0])
                difficult = int(items[1])
                x = int(items[2])
                y = int(items[3])
                w = int(items[4])
                h = int(items[5])
                rad = float(items[6])
                p = rbox_2_polygon(x, y, w, h, rad)
                rr, cc = polygon(p[1::2], p[0::2])
                rr = np.clip(rr, 0, im_height-1)
                cc = np.clip(cc, 0, im_width-1)
                mask = np.zeros([im_height, im_width], dtype=np.int32)
                mask[rr,cc] = 1
                obj = {
                    'name':'text',
                    'bndbox':{
                        'xmin':x,
                        'ymin':y,
                        'xmax':x+w,
                        'ymax':y+w,
                    },
                    'rotation':rad,
                    'mask':mask.flatten().tolist(),
                    'truncated': 0,
                    'pose': '',
                    'difficult': 1,
                }
                obj_list.append(obj)

        data = {
            'filename':fn,
            'object': obj_list,
        }

        print(os.path.join(data_dir, fn), im_width, im_height, len(data['object']))  ###
        if len(data['object']) >0:
            tf_example = dict_to_tf_example(data, data_dir, label_map_dict)
            writer.write(tf_example.SerializeToString())

    writer.close()

if __name__ == '__main__':
  tf.app.run()
