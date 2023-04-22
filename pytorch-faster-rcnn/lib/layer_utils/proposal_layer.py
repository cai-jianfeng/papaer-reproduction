# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from model.config import cfg
from model.bbox_transform import bbox_transform_inv, clip_boxes
from torchvision.ops import nms

import torch


def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride,
                   anchors, num_anchors):
    """A simplified version compared to fast/er RCNN
     For details please see the technical report
     rpn_cls_prob.shape: batch * h * w * (num_anchors * 2)
     rpn_bbox_pred.shape: batch * h * w * (num_anchors*4)
     im_info: (w, h, num_imgs)
     cfg_key: mode -> train/test
     anchors.shape: (num_anchors, 4)
  """
    if type(cfg_key) == bytes:
        cfg_key = cfg_key.decode('utf-8')
    pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
    post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
    nms_thresh = cfg[cfg_key].RPN_NMS_THRESH

    # Get the scores and bounding boxes
    scores = rpn_cls_prob[:, :, :, num_anchors:]  # 取每个anchor为前景的概率分数
    rpn_bbox_pred = rpn_bbox_pred.view((-1, 4))  # (batch * h * w * num_anchors) * 4
    scores = scores.contiguous().view(-1, 1)  # (batch * h * w * num_anchors * 2) * 1
    proposals = bbox_transform_inv(anchors, rpn_bbox_pred)  # ((batch * h * w * num_anchors) * 4)  --> (batch * h * w * num_anchors) == (num_anchors) ???
    proposals = clip_boxes(proposals, im_info[:2])  # ((batch * h * w * num_anchors) * 4)

    # Pick the top region proposals
    scores, order = scores.view(-1).sort(descending=True)  # sort 返回 value 和 indices
    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]
        scores = scores[:pre_nms_topN].view(-1, 1)
    proposals = proposals[order.data, :]

    # Non-maximal suppression
    keep = nms(proposals, scores.squeeze(1), nms_thresh)

    # Pick th top region proposals after NMS
    if post_nms_topN > 0:
        keep = keep[:post_nms_topN]
    proposals = proposals[keep, :]
    scores = scores[keep, ]  # scores[keep, ] = scores[keep, :]

    # Only support single image as input
    batch_inds = proposals.new_zeros(proposals.size(0), 1)  # batch_inds.shape = pre_nms_topN/post_nms_topN * 1
    blob = torch.cat((batch_inds, proposals), 1)  # pre_nms_topN/post_nms_topN * 5

    return blob, scores
