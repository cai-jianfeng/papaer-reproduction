# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from layer_utils.generate_anchors import generate_anchors


def generate_anchors_pre(height,
                         width,
                         feat_stride,
                         anchor_scales=(8, 16, 32),
                         anchor_ratios=(0.5, 1, 2)):
    """ A wrapper function to generate anchors given different scales
    Also return the number of anchors in variable 'length'
    对于一张宽高为 width x height 的图片的每个像素位置, 生成 len(anchor_scales) x len(anchor_ratios) 个的anchor, 每个anchor由左上+右下坐标表示,
    最后返回的anchors.shape = (num_anchors, 4); length = num_anchors
    """
    anchors = generate_anchors(
        ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))  # anchors.shape=(num_anchors, 4)
    A = anchors.shape[0]  # A = num_anchors
    shift_x = np.arange(0, width) * feat_stride
    shift_y = np.arange(0, height) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    # shift_x.ravel(): 将 shift_x 按行展平
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(),
                        shift_y.ravel())).transpose()
    K = shifts.shape[0]
    # width changes faster, so here it is H, W, C
    anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose(
        (1, 0, 2))  # anchors.shape = (K, A, 4)
    anchors = anchors.reshape((K * A, 4)).astype(np.float32, copy=False)
    length = np.int32(anchors.shape[0])

    return anchors, length