# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
try:
    import cPickle as pickle
except ImportError:
    import pickle
import os
import math

from utils.timer import Timer
from torchvision.ops import nms
from utils.blob import im_list_to_blob

from model.config import cfg, get_output_dir
from model.bbox_transform import clip_boxes, bbox_transform_inv

import torch


def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
    im (ndarray): a color image in BGR order
    Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    # 对每个像素都减去指定的均值
    im_orig -= cfg.PIXEL_MEANS  # cfg.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]]) shape = (1, 1, 3)

    im_shape = im_orig.shape  # (w, h, 3)
    # im_size_min/im_size_max: im 的 w, h 中的最小值/最大值
    im_size_min = np.min(im_shape[0:2])  # min(w, h)
    im_size_max = np.max(im_shape[0:2])  # max(w, h)

    processed_ims = []  # 表示对 im 进行的 scale 后的 im
    im_scale_factors = []  # 表示对 im 进行的 scale 比例

    for target_size in cfg.TEST.SCALES:  # (600,)
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:  # 1000
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(  # 将 im resize 成指定大小(最短边resize成600/最长边resize成1000)
            im_orig,
            None,
            None,
            fx=im_scale,
            fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    # 将processed_ims中的im变成统一大小(不满足条件的末尾补0)
    blob = im_list_to_blob(processed_ims)  # shape: (num_imgs, max_w, max_h, 3)

    return blob, np.array(im_scale_factors)  # shape: (num_imgs)


def _get_blobs(im):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {}
    # shape: (num_imgs, max_w, max_h, 3); shape: (num_imgs)
    blobs['data'], im_scale_factors = _get_image_blob(im)  # 将 im 先 resize 成指定的不同大小(根据 cfg.TEST.SCALES 和 cfg.TEST.MAX_SIZE), 然后统一大小(所有图片的最大宽和最大高)

    return blobs, im_scale_factors


def _clip_boxes(boxes, im_shape):
    """Clip boxes to image boundaries."""
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    return boxes


def _rescale_boxes(boxes, inds, scales):
    """Rescale boxes according to image rescaling."""
    for i in range(boxes.shape[0]):
        boxes[i, :] = boxes[i, :] / scales[int(inds[i])]

    return boxes


def im_detect(net, im):
    # im.shape: (w, h, 3)
    # blobs['data'].shape: (num_imgs, max_w, max_h, 3); im_scales.shape: (num_imgs)
    blobs, im_scales = _get_blobs(im)
    assert len(im_scales) == 1, "Only single-image batch implemented"

    im_blob = blobs['data']  # (num_imgs -> 同一张图片不同缩放大小的结果, w, h, 3)
    blobs['im_info'] = np.array(  # im_info = (max_w, max_h, num_imgs)
        [im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)
    # scores.shape = (num_rois, num_class)
    # bbox_pred.shape = (num_rois, num_class * 4)
    # rois.shape = pre_nms_topN/post_nms_topN * 5 -> 第一列是类别(全是前景类, 即全为0)
    _, scores, bbox_pred, rois = net.test_image(blobs['data'],
                                                blobs['im_info'])

    boxes = rois[:, 1:5] / im_scales[0]  # 将roi还原回原始图片的对应大小(原始图片经过im_scales缩放后进入net, 得到的roi也是缩放后图像对应位置的box)
    scores = np.reshape(scores, [scores.shape[0], -1])
    bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred
        # 将原始的boxes和预测得到的偏移deltas转化为最终加上偏移deltas后的预测位置pred_boxes
        pred_boxes = bbox_transform_inv(
            torch.from_numpy(boxes), torch.from_numpy(box_deltas)).numpy()
        # 将 pred_boxes归一化到合理的范围, 即保证pred_boxes在图片内(x1不会比0小, x2不会比宽大; y1, y2同理)
        pred_boxes = _clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    return scores, pred_boxes


def test_net(net, imdb, weights_filename, max_per_image=100, thresh=0.):
    np.random.seed(cfg.RNG_SEED)
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #  all_boxes[cls][image] = N x 5 array of detections in
    #  (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]

    output_dir = get_output_dir(imdb, weights_filename)
    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}

    for i in range(num_images):
        im = cv2.imread(imdb.image_path_at(i))

        _t['im_detect'].tic()
        scores, boxes = im_detect(net, im)
        _t['im_detect'].toc()

        _t['misc'].tic()

        # skip j = 0, because it's the background class
        for j in range(1, imdb.num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
              .astype(np.float32, copy=False)
            keep = nms(
                torch.from_numpy(cls_boxes), torch.from_numpy(cls_scores),
                cfg.TEST.NMS).numpy() if cls_dets.size > 0 else []
            cls_dets = cls_dets[keep, :]
            all_boxes[j][i] = cls_dets

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack(
                [all_boxes[j][i][:, -1] for j in range(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        _t['misc'].toc()

        print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
            .format(i + 1, num_images, _t['im_detect'].average_time(),
                _t['misc'].average_time()))

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir)
