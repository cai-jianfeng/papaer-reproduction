# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

# array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])


def generate_anchors(base_size=16,
                     ratios=[0.5, 1, 2],
                     scales=2**np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """
    # anchor_scales=(8, 16, 32),
    # anchor_ratios=(0.5, 1, 2)
    base_anchor = np.array([1, 1, base_size, base_size]) - 1  # (0, 0, 15, 15)
    ratio_anchors = _ratio_enum(base_anchor, ratios)  # shape = (num_anchors = 3, 4)
    anchors = np.vstack([
        _scale_enum(ratio_anchors[i, :], scales)
        for i in range(ratio_anchors.shape[0])
    ])
    return anchors  # shape = (num_anchors=9, 4)


def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
  Given a vector of widths (ws) and heights (hs) around a center
  (x_ctr, y_ctr), output a set of anchors (windows).
  """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)))
    return anchors  # shape = num_anchor, 4


def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    给定初始的anchor大小:(0, 0, 15, 15)和缩放比例ratio:(0.5, 1, 2） -> 返回缩放后的anchor(宽高的缩放比例不同, 使得缩放后端anchor的面积和原始的正方形框差不多: w = ((w x h) / scale)^0.5, w x scale)
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h  # 16 x 16
    size_ratios = size / ratios  # (512, 256, 128)
    ws = np.round(np.sqrt(size_ratios))  # 23, 16, 11  ->  np.round: 四舍五入
    hs = np.round(ws * ratios)  # 12, 16, 22 -> ws 和 hs 一一对应, 一共是3个anchor
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors  # shape = (num_anchors = 3, 4)


def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """
    # anchor.shape = (4, )
    # scales = (8, 16, 32)
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors  # shape = (num_anchors = 3, 4)


if __name__ == '__main__':
    import time

    t = time.time()
    a = generate_anchors()
    print(time.time() - t)
    print(a)
    from IPython import embed

    embed()
