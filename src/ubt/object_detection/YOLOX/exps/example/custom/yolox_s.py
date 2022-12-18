#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

import torch.nn as nn
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.25 # def 0.5
        self.test_size = (416, 416)
        self.input_size = (416, 416)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "datasets/multitracker"
        self.train_ann = "train2017.json"
        self.val_ann = "test2017.json"

        self.num_classes = 2

        self.max_epoch = 20000
        self.data_num_workers = 8
        self.eval_interval = 40
        #self.basic_lr_per_img *= 2.0
        #self.warmup_epochs = 500
        #self.warmup_lr = 5e-3
        #self.no_aug_epochs = 3000
        self.hsv_prob = 0

        self.mixup_prob = 0
        self.mosaic_prob = 0
        self.shear = 0.0
        self.degrees = 0
        self.translate = 0 
        #self.enable_mixup = False

    
    def get_model(self, sublinear=False):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
            in_channels = [256, 512, 1024]
            # NANO model use depthwise = True, which is main difference.
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, depthwise=True)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, depthwise=True)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
