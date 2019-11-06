import torch
import torch.nn as nn
from losses import iou
from anchors import Anchors
from nms import gpu_nms

class AnchorTargetCreator():

    def __init__(self, n_sample=256, positive_iou_threshold=0.7,
                 negative_iou_threshold=0.3, positive_ratio=0.5):

        self.n_sample = n_sample
        self.positive_iou_threshold = positive_iou_threshold
        self.negative_iou_threshold = negative_iou_threshold
        self.positive_ratio = positive_ratio

    # def forward(self, threshold, img, annotations):
    #
    #     batch_size = annotations.shape[0]
    #
    #     anchors = self.anchor(img)
    #     anchor = anchors[0, :, :]
    #
    #     for j in range(batch_size):
    #         bbox_annotation = annotations[j, :, :]
    #         bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]
    #
    #         if bbox_annotation.shape[0] == 0:
    #
    #             continue
    #
    #     return positive_sample, negative_sample

    def __call__(self, bbox, anchor, img_size):
        pass