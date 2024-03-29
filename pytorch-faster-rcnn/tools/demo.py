#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect

from torchvision.ops import nms

from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

import torch

CLASSES = ('__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
           'tvmonitor')

NETS = {
    'vgg16': ('vgg16_faster_rcnn_iter_%d.pth', ),
    'res101': ('res101_faster_rcnn_iter_%d.pth', )
}

DATASETS = {
    'pascal_voc': ('voc_2007_trainval', ),
    'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval', )
}

'''
    对于opencv和detectron2的faster-rcnn检测: 左上角为原点, 即位置为(0, 0). 
        cv2.rectangle 的参数为: 左上角坐标, 右下角坐标 
    对于matplotlib: 也是左上角为原点, 
        matplotlin.pyplot.Rectangle 的参数为 左上角坐标, width, height
'''
def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    # dets shape: [num_box, 5] (5 -> x1,y1,x2,y2,score)
    # np.where: 没有指定返回结果, 只有查找条件的情况下 -> 返回满足条件的位置(2个元组); 第一个元组是行, 第二个元组是列
    inds = np.where(dets[:, -1] >= thresh)[0]  # 这里我们是需要满足score大于阈值(thresh)的所有框的index, 所以需要行
    if len(inds) == 0:  # 所有的预测框的分数都没超过阈值
        return

    im = im[:, :, (2, 1, 0)]  # BGR -> RGB
    # plt.subplots: 创建子图(一次性全部创建)
    # fig: 图窗; ax: 子图的坐标区
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')  # aspect: 表示横纵坐标的比例 —> equal表示正方形, 还可设置为 auto
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(  # 在 ax 的坐标区添加子形状
            plt.Rectangle((bbox[0], bbox[1]),  # plt.Rectangle: 输入左上角坐标 + w + h
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1],
                          fill=False,  # 是否将矩形填充
                          edgecolor='red',  # 边框颜色
                          linewidth=3.5))  # 边框线条粗细
        ax.text(  # 在 ax 的坐标区添加注释
            bbox[0],
            bbox[1] - 2,
            '{:s} {:.3f}'.format(class_name, score),
            bbox=dict(facecolor='blue', alpha=0.5),
            fontsize=14,
            color='white')

    ax.set_title(  # 设置子坐标区的标题
        ('{} detections with '
         'p({} | box) >= {:.1f}').format(class_name, class_name, thresh),
        fontsize=14)
    plt.axis('off')
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域 -> 检查坐标轴标签、刻度标签以及标题的部分
    plt.draw()


def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)  # score shape: (num_boxes, len(CLASSES)); boxes shape: (num_boxes, len(CLASSES) * 4)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(
        timer.total_time(), boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]  # (num_boxes, 4)
        cls_scores = scores[:, cls_ind]  # (num_boxes, )
        # np.newaxis: 相当于 torch 的 unsqueeze, 在增加指定维度
        # np.hstack: 将两个 ndarray 沿着 列 进行拼接
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)  # (num_boxes, 5)
        keep = nms(
            torch.from_numpy(cls_boxes), torch.from_numpy(cls_scores),
            NMS_THRESH)  # nms: 不分类别, 对于IOU超过阈值的只保留分数高的那个(不同于仅在同一类之间进行nms -> batched_nms): 参数 boxes(shape: num_boxes, 4), scores(shape: num_boxes, ), nms阈值
        # nms 返回保留下来的序号(从0开始) -> tensor
        dets = dets[keep.numpy(), :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Tensorflow Faster R-CNN demo')
    parser.add_argument(
        '--net',
        dest='demo_net',
        help='Network to use [vgg16 res101]',
        choices=NETS.keys(),
        default='res101')
    parser.add_argument(
        '--dataset',
        dest='dataset',
        help='Trained dataset [pascal_voc pascal_voc_0712]',
        choices=DATASETS.keys(),
        default='pascal_voc_0712')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    # os.path.join: 可以拼接任意的字符串为路径(每个字符串之间用 \\ 连接)
    saved_model = os.path.join(
        'output', demonet, DATASETS[dataset][0], 'default',
        NETS[demonet][0] % (70000 if dataset == 'pascal_voc' else 110000))

    if not os.path.isfile(saved_model):
        raise IOError(
            ('{:s} not found.\nDid you download the proper networks from '
             'our server and place them properly?').format(saved_model))

    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    # 创建模型, 包括导入 torchvision 内置的 vgg16 作为主干网络和 自定义的 rpn(包括 rpn 主干网络, rpn 的分类头和box偏移预测量) 与 最终的分类头和box偏移预测量
    net.create_architecture(21, tag='default', anchor_scales=[8, 16, 32])

    net.load_state_dict(
        # Load all tensors onto the CPU, using a function
        torch.load(saved_model, map_location=lambda storage, loc: storage))

    net.eval()
    if not torch.cuda.is_available():
        net._device = 'cpu'
    net.to(net._device)

    print('Loaded network {:s}'.format(saved_model))

    im_names = [
        '000456.jpg', '000542.jpg', '001150.jpg', '001763.jpg', '004545.jpg'
    ]
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        demo(net, im_name)

    plt.show()
