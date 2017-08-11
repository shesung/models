#!/usr/bin/env python

r"""Convert text localization datasets to TFRecord.

Example usage:
    python create_text_localization_tf_record.py \
        --output_path=/path/to/textloc.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os
import time
import math

import numpy as np
from skimage.draw import polygon

from lxml import etree
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


flags = tf.app.flags

flags.DEFINE_string('set', 'MSRA-TD500_train', 'dataset to pack, sperated by comma')
flags.DEFINE_string('output_path', './text_loc.record', 'Path to output TFRecord')
flags.DEFINE_integer('min_size', 640, 'Size of resized shorter edge')

FLAGS = flags.FLAGS


SETS = ['MSRA-TD500_train', 'MSRA-TD500_test', 
        'ICDAR2015_train', 
        'ICDAR2013_train', 'ICDAR2013_test']

def read_multiple_dataset(data_str):
    anno = dict()
    for dataset in data_str.strip().split(','):
        if dataset not in SETS:
            raise ValueError('set must be in : {}'.format(SETS))
        anno.update(read_anno(dataset))
    return anno


def read_anno(dataset):
    if dataset == 'MSRA-TD500_train':
        return read_msra('/world/data-c7/censhusheng/data/MSRA-TD500/train')
    elif dataset == 'MSRA-TD500_test':
        return read_msra('/world/data-c7/censhusheng/data/MSRA-TD500/test')
    elif dataset == 'ICDAR2015_train':
        return read_icdar_2015('/world/data-c7/censhusheng/data/icdar2015-Incidental_Scene_Text/train_images',
                               '/world/data-c7/censhusheng/data/icdar2015-Incidental_Scene_Text/train_gt')
    elif dataset == 'ICDAR2013_train':
        return read_icdar_2013('/world/data-c7/censhusheng/data/icdar2013-Focused_Scene_Text/train_images',
                               '/world/data-c7/censhusheng/data/icdar2013-Focused_Scene_Text/train_gt')
    elif dataset == 'ICDAR2013_test':
        return read_icdar_2013('/world/data-c7/censhusheng/data/icdar2013-Focused_Scene_Text/test_images',
                               '/world/data-c7/censhusheng/data/icdar2013-Focused_Scene_Text/test_gt')
    else:
        return dict()
    
def read_icdar_2015(image_dir, gt_dir):
    anno_dict = dict()
    for fn in os.listdir(image_dir):
        gt_fn = 'gt_' + os.path.splitext(fn)[0] + '.txt'
        obj_list = []
        with open(os.path.join(gt_dir, gt_fn)) as f_gt:
            for line in f_gt.readlines():
                line = line.decode("utf-8-sig").encode("utf-8")
                items = line.strip().split(',')
                poly = [int(x) for x in items[:8]]
                rbox = polygon_2_rbox(poly)
                obj_list.append((rbox, poly))
        anno_dict[os.path.join(image_dir, fn)] = obj_list
    return anno_dict
    
def read_icdar_2013(image_dir, gt_dir):
    anno_dict = dict()
    for fn in os.listdir(image_dir):
        gt_fn = 'gt_' + os.path.splitext(fn)[0] + '.txt'
        obj_list = []
        with open(os.path.join(gt_dir, gt_fn)) as f_gt:
            for line in f_gt.readlines():
                line = line.decode("utf-8-sig").encode("utf-8")
                items = line.strip().split(',')
                if len(items) < 5:
                    items = line.strip().split(' ')
                xmin, ymin, xmax, ymax = [int(x) for x in items[:4]]
                xc = (xmin+xmax)/2
                yc = (ymin+ymax)/2
                w = xmax - xmin
                h = ymax - ymin
                rbox = [xc, yc, w, h, 0]
                poly = rbox_2_polygon(*rbox)
                obj_list.append((rbox, poly))
        anno_dict[os.path.join(image_dir, fn)] = obj_list
    return anno_dict


def read_msra(data_dir):
    anno_dict = dict()
    for fn in os.listdir(data_dir):
        if os.path.splitext(fn)[1] == '.gt':
            continue

        gt_fn = os.path.splitext(fn)[0] + '.gt'
        obj_list = []
        with open(os.path.join(data_dir, gt_fn)) as f_gt:
            for line in f_gt.readlines():
                items = line.strip().split()
                index = int(items[0])
                difficult = int(items[1])
                x, y, w, h, rad = [float(x) for x in items[2:7]]
                rbox = [x+w/2, y+w/2, w, h, rad]
                poly = rbox_2_polygon(*rbox)
                obj_list.append((rbox, poly))
        anno_dict[os.path.join(data_dir, fn)] = obj_list
    return anno_dict


def dict_to_tf_example(data, label_map_dict):
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
  print('mask encode:', mask.shape, '->', len(encoded_mask)) ###
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



#####################################
########### geo func ###############
#####################################
def rbox_2_polygon(xc, yc, w, h, rad):
    w2 = w/2
    h2 = h/2
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


def polygon_2_rbox(poly):
    x1, y1, x2, y2, x3, y3, x4, y4 = poly
    theta = math.atan2(y2-y1, x2-x1)
    pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32)
    pts[:, 0] = pts[:, 0] - x1
    pts[:, 1] = pts[:, 1] - y1
    m_rot = np.array([[np.cos(theta), np.sin(theta)],
                      [-np.sin(theta), np.cos(theta)]], dtype=np.float32)
    pts_ = np.dot(m_rot, pts.T)
    w = np.amax(pts_[0,:]) - np.amin(pts_[0,:])
    h = np.amax(pts_[1,:]) - np.amin(pts_[1,:])
    xc_ = (np.amax(pts_[0,:]) + np.amin(pts_[0,:]))/2
    yc_ = (np.amax(pts_[1,:]) + np.amin(pts_[1,:]))/2
    xc = x1 + xc_ * np.cos(theta) - yc_ * np.sin(theta)
    yc = y1 + xc_ * np.sin(theta) + yc_ * np.cos(theta)
    return xc, yc, w, h, theta


def shrink_quadrangle(poly, t=0.3):
    x1, y1, x2, y2, x3, y3, x4, y4 = poly
    
    d1 = max(2, math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)))
    d2 = max(2, math.sqrt((x3 - x2) * (x3 - x2) + (y3 - y2) * (y3 - y2)))
    d3 = max(2, math.sqrt((x4 - x3) * (x4 - x3) + (y4 - y3) * (y4 - y3)))
    d4 = max(2, math.sqrt((x1 - x4) * (x1 - x4) + (y1 - y4) * (y1 - y4)))
    min_d = min([d1, d2, d3, d4])

    x1_ = x1 + t * (x2 - x1) * min_d / d1
    y1_ = y1 + t * (y2 - y1) * min_d / d1
    x2_ = x2 + t * (x1 - x2) * min_d / d1
    y2_ = y2 + t * (y1 - y2) * min_d / d1
    x3_ = x3 + t * (x4 - x3) * min_d / d3
    y3_ = y3 + t * (y4 - y4) * min_d / d3
    x4_ = x4 + t * (x3 - x4) * min_d / d3
    y4_ = y4 + t * (y3 - y4) * min_d / d3
    x1, y1, x2, y2, x3, y3, x4, y4 = x1_, y1_, x2_, y2_, x3_, y3_, x4_, y4_

    x1_ = x1 + t * (x4 - x1) * min_d / d4
    y1_ = y1 + t * (y4 - y1) * min_d / d4
    x4_ = x4 + t * (x1 - x4) * min_d / d4
    y4_ = y4 + t * (y1 - y4) * min_d / d4
    x2_ = x2 + t * (x3 - x2) * min_d / d2
    y2_ = y2 + t * (y3 - y2) * min_d / d2
    x3_ = x3 + t * (x2 - x3) * min_d / d2
    y3_ = y3 + t * (y2 - y3) * min_d / d2

    return [x1_, y1_, x2_, y2_, x3_, y3_, x4_, y4_]


#####################################
########### util func ###############
#####################################

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
        new_height = int(min_size)
        new_width = int(width * min_size / height)
        if new_width % 32 != 0:
            new_width -= new_width%32
    else:
        new_width = int(min_size)
        new_height = int(height * min_size / width)
        if new_height % 32 != 0:
            new_height -= new_height%32
    print('resize:', (width, height), '->', (new_width, new_height)) ###
    return image.resize((new_width, new_height))



def main(_):

    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    label_map_dict = label_map_util.get_label_map_dict('object_detection/data/text_label_map.pbtxt')


    anno_dict = read_multiple_dataset(FLAGS.set)
    cnt = 0
    for image_fn in anno_dict:
        print(image_fn) ###
        
        t0 = time.time()
        im = PIL.Image.open(image_fn)
        
        # resize
        width, height = im.size
        im = resize(im, FLAGS.min_size)
        new_width, new_height = im.size
        ratio_x = float(new_width) / width
        ratio_y = float(new_height) / height
        
        obj_list = []
        for rbox, poly in anno_dict[image_fn]:
            p = shrink_quadrangle(poly, 0.3)
            xc, yc, w, h, rad = rbox
            # scale rbox
            xc *= ratio_x
            yc *= ratio_y
            w  *= ratio_x
            h  *= ratio_y
            # scale poly
            r = np.array(p[1::2]) * ratio_y
            c = np.array(p[0::2]) * ratio_x
            rr, cc = polygon(r, c)
            rr = np.clip(rr, 0, new_height-1)
            cc = np.clip(cc, 0, new_width-1)
            
            mask = np.zeros([new_height, new_width], dtype=np.int32)
            mask[rr,cc] = 1
            
            obj = {
                'name':'text',
                'bndbox':{
                    'xmin':int(xc - w/2),
                    'ymin':int(yc - h/2),
                    'xmax':int(xc + w/2),
                    'ymax':int(yc + h/2),
                },
                'rotation':rad,
                'mask': mask,
                'truncated': 0,
                'pose': '',
                'difficult': 0,
            }
            obj_list.append(obj)
            
        data = {
            'filename':image_fn,
            'object': obj_list,
            'image': im
        }

        if len(data['object']) >0:
            tf_example = dict_to_tf_example(data, label_map_dict)
            writer.write(tf_example.SerializeToString())
            cnt += 1

        print('boxes:%d'%(len(data['object'])), 
              'time:%f'%(time.time() - t0),
              '\n----------------------------')  ###

    writer.close()
    print('total image:', cnt) ###


if __name__ == '__main__':
    tf.app.run()
