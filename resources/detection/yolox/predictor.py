#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import numpy as np

import torch

from .yolox.data.data_augment import ValTransform
from .yolox.utils import postprocess


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        fp16=False,
    ):
        self.model = model
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = "gpu" if torch.cuda.is_available() else "cpu"
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=False)

    def inference(self, img):
        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16
   
        with torch.no_grad():
            outputs = self.model(img)
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            
        outputs = outputs[0].cpu().detach().numpy()
        outputs = outputs[outputs[:, 6] == 0]        
        bboxes = outputs[:, 0:4]
        bboxes /= ratio
        bboxes[:, 2] -= bboxes[:, 0]
        bboxes[:, 3] -= bboxes[:, 1]
        scores = outputs[:, 4] * outputs[:, 5]
        h, _ = bboxes.shape
        outputs = np.c_[-np.ones((h, 2)), bboxes, scores, -np.ones((h, 3))]
        
        return outputs