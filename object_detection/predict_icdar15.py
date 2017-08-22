import numpy as np
import os
import sys
import time

import numpy as np

import tensorflow as tf
print tf.__file__ ###

from collections import defaultdict

from PIL import Image

sys.path.append("..")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from utils import label_map_util

PATH_TO_CKPT = '/world/data-c9/censhusheng/train_tf_text_loc/inference_graph-2.pb'

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

PATH_TO_LABELS = 'data/text_label_map.pbtxt'
NUM_CLASSES = 1

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_indexs = label_map_util.create_category_index(categories)


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
    print 'resize:', (width, height), '->', (new_width, new_height) ###
    return image.resize((new_width, new_height), Image.BILINEAR)



def load_image_into_numpy_array(input_image, min_size=640):
    image = resize(input_image, min_size)
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


import math
def rbox_2_polygon(xc, yc, w, h, rad):
    if w <1 or h <1:
        return None
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



sys.path.append(os.path.expanduser('~/github/EAST/'))
import lanms
print lanms



data_dir = '/world/data-c7/censhusheng/data/icdar2015-Incidental_Scene_Text/test_images/'
output_dir = '/world/data-c7/censhusheng/data/icdar2015-Incidental_Scene_Text/script_test/test'
config = tf.ConfigProto(
        inter_op_parallelism_threads=2,
        intra_op_parallelism_threads=2,
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction = 0.5
            )
        )

with detection_graph.as_default():
    with tf.Session(graph=detection_graph, config=config) as sess:
        for i, fn in enumerate(os.listdir(data_dir)):
            print '\n', i, os.path.join(data_dir, fn) ###
            t_ = time.time()
            image = Image.open(os.path.join(data_dir, fn))
            width, height = image.size
            image_np = load_image_into_numpy_array(image, 736)
            new_width, new_height = image.size
            ratio_x = float(width)/image_np.shape[1]
            ratio_y = float(height)/image_np.shape[0]
            print 'resize time:', time.time() - t_ ###

            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # Actual detection.
            t0 = time.time()
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            print 'net time:', time.time() - t0, 'num_detections:', num_detections ###

            # convert rbox to polygon
            t0 = time.time()
            boxes = np.squeeze(boxes)
            yc = (boxes[:,0] + boxes[:,2]) / 2
            xc = (boxes[:,1] + boxes[:,3]) / 2
            h = boxes[:,2] - boxes[:,0]
            w = boxes[:,3] - boxes[:,1]
            scores = np.squeeze(scores)
            polys = []
            score_thresh = 0.9
            for i in range(num_detections[0]):
                if scores[i] < score_thresh:
                    continue
                p = rbox_2_polygon(xc[i], yc[i], w[i], h[i], boxes[i,4])
                if p is not None:
                    polys.append(p + [scores[i]])
            polys = np.array(polys, dtype=np.float32)
            print 'convert time:', time.time() - t0 ###

            # lanms
            nms_thresh = 0.2
            t0 = time.time()
            polys = lanms.merge_quadrangle_n9(polys, nms_thresh)
            nms_keep_thresh = 5.0
            if polys.shape[0] > 0:
                remain_index = np.where(polys[:,8] > nms_keep_thresh)[0]
                polys = polys[remain_index]
            print 'nms_time:', time.time() - t0, 'boxes:', polys.shape[0]  ###

            # save results
            res_fn = 'res_' + os.path.splitext(fn)[0] + '.txt'
            with open(os.path.join(output_dir, res_fn), 'w') as f_res:
                for p in polys:
                    f_res.write("{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d}\r\n".format(
                        int(p[0] * ratio_x),
                        int(p[1] * ratio_y),
                        int(p[2] * ratio_x),
                        int(p[3] * ratio_y),
                        int(p[4] * ratio_x),
                        int(p[5] * ratio_y),
                        int(p[6] * ratio_x),
                        int(p[7] * ratio_y),
                    ))
            print 'total time:', time.time() - t_
