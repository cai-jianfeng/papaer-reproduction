# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import time
import torch


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self._total_time = {}
        self._calls = {}
        self._start_time = {}
        self._diff = {}
        self._average_time = {}

    def tic(self, name='default'):
        # using time.time instead of time.clock because time time.clock does not normalize for multithreading
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # 等待当前GPU设备上所有流中的所有核心完成, 在GPU的任务都完成后才会执行后续的代码
        self._start_time[name] = time.time()

    def toc(self, name='default', average=True):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._diff[name] = time.time() - self._start_time[name]
        self._total_time[name] = self._total_time.get(name,
                                                      0.) + self._diff[name]  # key=name的已经存在的时间+本次消耗的时间
        self._calls[name] = self._calls.get(name, 0) + 1  # key=name的调用tic的次数
        self._average_time[name] = self._total_time[name] / self._calls[name]  # key=name的平均消耗时间
        if average:
            return self._average_time[name]
        else:
            return self._diff[name]

    def average_time(self, name='default'):
        return self._average_time[name]

    def total_time(self, name='default'):
        return self._total_time[name]


timer = Timer()
