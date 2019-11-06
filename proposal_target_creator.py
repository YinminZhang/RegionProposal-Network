import torch
import torch.nn as nn
import numpy as np
from losses import iou
from anchors import Anchors
from nms import gpu_nms
from bbox import bbox2loc, loc2bbox

class ProposalTargetCreator:
    def __init__(self, num_sample=128, positive_ratio=0.25,
                 negative_iou_threshold_high=0.5,
                 negative_iou_threshold_low=0.0):
        self.num_sample = num_sample
        self.positive_ratio = positive_ratio
        self.negative_iou_threshold_high = negative_iou_threshold_high
        self.negative_iou_threshold_low = negative_iou_threshold_low

    def __call__(self, roi, bbox, label,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        pass