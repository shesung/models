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
import time

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
flags.DEFINE_integer('min_size', 800, 'Size of resized shorter edge')

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
  encoded_jpg_io = io.BytesIO()
  image = data['image']
  image.save(encoded_jpg_io, "JPEG", quality=80)
  encoded_jpg = encoded_jpg_io.getvalue()
  key = hashlib.sha256(encoded_jpg).hexdigest()

  width, height = image.size

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  rotation = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  masks = []
  difficult_obj = []
  for obj in data['object']:
    difficult = bool(int(obj['difficult']))
    difficult_obj.append(int(difficult))

    xmin.append(float(obj['bndbox']['xmin']) / width)
    ymin.append(float(obj['bndbox']['ymin']) / height)
    xmax.append(float(obj['bndbox']['xmax']) / width)
    ymax.append(float(obj['bndbox']['ymax']) / height)
    rotation.append(float(obj['rotation']))
    masks.append(obj['mask'])
    classes_text.append(obj['name'].encode('utf8'))
    classes.append(label_map_dict[obj['name']])
    truncated.append(int(obj['truncated']))
    poses.append(obj['pose'].encode('utf8'))

  mask = np.stack(masks)
  encoded_mask = pn_encode(mask.flatten()).tolist()
  mask_length = len(encoded_mask)
  print('mask:', mask.shape, '->', len(encoded_mask)) ###
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
      'image/segmentation/object': dataset_util.int64_list_feature(encoded_mask),
      'image/segmentation/length': dataset_util.int64_feature(mask_length),
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

def pn_encode(x):
    '''
    x: 1-D numpy array of 0 or 1
    '''
    x = x * 2 - 1
    x = np.concatenate([[-1 *x[0]], x, [-1 *x[-1]]])
    pd = np.where(np.diff(x) != 0)[0]
    d = pd[1:] - pd[:-1]
    c = np.multiply(d, x[pd[1:]])
    return c.astype(np.int32)

def resize(image, min_size):
    width, height = image.size
    if height < width:
        new_height = int(800)
        new_width = int(width * 800 / height)
        if new_width % 32 != 0:
            new_width -= new_width%32
    else:
        new_width = int(800)
        new_height = int(height * 800 / width)
        if new_height % 32 != 0:
            new_height -= new_height%32
    return image.resize((new_width, new_height))

def main(_):
    if FLAGS.set not in SETS:
        raise ValueError('set must be in : {}'.format(SETS))

    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    label_map_dict = label_map_util.get_label_map_dict('object_detection/data/text_label_map.pbtxt')

    data_dir = FLAGS.data_dir
    for fn in os.listdir(data_dir):
        if os.path.splitext(fn)[1] == '.gt':
            continue

        t0 = time.time()
        im = PIL.Image.open(os.path.join(data_dir, fn))

        # resize
        width, height = im.size
        im = resize(im, FLAGS.min_size)
        new_width, new_height = im.size
        radio_x = float(new_width) / width
        radio_y = float(new_height) / height

        gt_fn = os.path.splitext(fn)[0] + '.gt'
        obj_list = []
        with open(os.path.join(data_dir, gt_fn)) as f_gt:
            for line in f_gt.readlines():
                #line = line.decode("utf-8").encode("utf-8")
                items = line.strip().split()
                index = int(items[0])
                difficult = int(items[1])
                x = float(items[2])*radio_x
                y = float(items[3])*radio_y
                w = float(items[4])*radio_x
                h = float(items[5])*radio_y
                rad = float(items[6])

                x1, y1, x2, y2, x3, y3, x4, y4 = rbox_2_polygon(x, y, w, h, rad)
                t = 0.25

                x1_ = x1 + t * (x2 - x1)
                y1_ = y1 + t * (y2 - y1)
                x2_ = x2 + t * (x1 - x2)
                y2_ = y2 + t * (y1 - y2)
                x3_ = x3 + t * (x4 - x3)
                y3_ = y3 + t * (y4 - y4)
                x4_ = x4 + t * (x3 - x4)
                y4_ = y4 + t * (y3 - y4)
                x1, y1, x2, y2, x3, y3, x4, y4 = x1_, y1_, x2_, y2_, x3_, y3_, x4_, y4_

                x1_ = x1 + t * (x4 - x1)
                y1_ = y1 + t * (y4 - y1)
                x4_ = x4 + t * (x1 - x4)
                y4_ = y4 + t * (y1 - y4)
                x2_ = x2 + t * (x3 - x2)
                y2_ = y2 + t * (y3 - y2)
                x3_ = x3 + t * (x2 - x3)
                y3_ = y3 + t * (y2 - y3)

                p = [x1_, y1_, x2_, y2_, x3_, y3_, x4_, y4_]
                rr, cc = polygon(p[1::2], p[0::2])
                rr = np.clip(rr, 0, new_height-1)
                cc = np.clip(cc, 0, new_width-1)
                mask = np.zeros([new_height, new_width], dtype=np.int32)
                mask[rr,cc] = 1
                obj = {
                    'name':'text',
                    'bndbox':{
                        'xmin':x,
                        'ymin':y,
                        'xmax':x+w,
                        'ymax':y+h,
                    },
                    'rotation':rad,
                    'mask': mask,
                    'truncated': 0,
                    'pose': '',
                    'difficult': 1,
                }
                obj_list.append(obj)

        print('parse:', time.time() -t0) ###
        data = {
            'filename':fn,
            'object': obj_list,
            'image': im
        }

        if len(data['object']) >0:
            t0 = time.time()
            tf_example = dict_to_tf_example(data, data_dir, label_map_dict)
            writer.write(tf_example.SerializeToString())
            print('write', time.time() - t0)
        print(os.path.join(data_dir, fn), (new_width, new_height), len(data['object']))  ###
        print('------------') ###

    writer.close()

if __name__ == '__main__':
  tf.app.run()
