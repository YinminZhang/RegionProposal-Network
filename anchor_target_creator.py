import torch
import torch.nn as nn
import numpy as np
from losses import iou
from anchors import Anchors
from nms import gpu_nms
from bbox import bbox2loc, loc2bbox

class AnchorTargetCreator():

    def __init__(self, n_sample=256, positive_iou_threshold=0.7,
                 negative_iou_threshold=0.3, positive_ratio=0.5):

        self.n_sample = n_sample
        self.positive_iou_threshold = positive_iou_threshold
        self.negative_iou_threshold = negative_iou_threshold
        self.positive_ratio = positive_ratio

    def __call__(self, bbox, anchors, img_size):
        height, width = img_size

        num_anchors = anchors.shape[0]
        anchors[:, 0] = torch.clamp_min(anchors[:, 0], 0)
        anchors[:, 1] = torch.clamp_min(anchors[:, 1], 1)
        anchors[:, 2] = torch.clamp_max(anchors[:, 2], height)
        anchors[:, 3] = torch.clamp_max(anchors[:, 3], width)

        targets = torch.ones(num_anchors) * -1
        targets = targets.cuda()

        IoU = iou(anchors, bbox)
        IoU_max, IoU_argmax = torch.max(IoU, dim=1)
        gt_IoU_max, gt_Iou_argmax = torch.max(IoU, dim=0)

        targets[IoU_max < self.negative_iou_threshold] = 0

        targets[gt_Iou_argmax] = 1

        targets[IoU_max > self.positive_iou_threshold] = 1

        # subsample positive labels if too many labels are positive
        num_positive = self.positive_ratio * num_anchors
        positive_index = (targets == 1).nonzero().cpu().numpy()

        if len(positive_index) > num_positive:
            disable_index =  np.random.choice(positive_index, size=(len(positive_index) - num_positive), replace=False)
            disable_index = torch.from_numpy(disable_index).cuda()
            targets[disable_index] = -1

        # subsample negative labels if too many labels are negative
        num_negative = self.n_sample - (targets == 1).sum()
        neg_index = (targets == 0).nonzero().cpu().numpy()
        if len(neg_index) > num_negative:
            disable_index = np.random.choice(neg_index, size=(len(neg_index) - num_negative), replace=False)
            disable_index = torch.from_numpy(disable_index).cuda()
            targets[disable_index] = -1

        loc = bbox2loc(anchors, bbox[IoU_argmax])

        return loc, targets


class ProposalCreator:
    def __init__(self, parent_model, nms_threshold=0.7,
                 n_train_pre_nms=12000, n_train_post_nms=2000,
                 n_test_pre_nms = 6000, n_test_post_nms=300,
                 min_size=16):
        self.parent_model = parent_model
        self.nms_threshold = nms_threshold
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score,
                 anchor, img_size, scale=1.):
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_pre_nms

        roi = loc2bbox(anchor, loc)
        roi[:, 0] = torch.clamp_min(roi[:, 0], 0)
        roi[:, 1] = torch.clamp_min(roi[:, 1], 0)
        roi[:, 2] = torch.clamp_max(roi[:, 2], img_size[0])
        roi[:, 3] = torch.clamp_max(roi[:, 3], img_size[1])

        min_size = self.min_size
        height = roi[:, 2] -roi[:, 0]
        width = roi[:, 3] - roi[:, 1]
        keep = (height > min_size) & (width > min_size)
        roi = roi[keep, :]
        score = score[keep]

        if n_pre_nms > score.shape[0]:
            order = score.sort(0, descending=True)
            roi = roi[order[1][:, n_pre_nms], :]

        keep_index, _ = gpu_nms(torch.stack((roi, score), dim=1), self.nms_threshold)
        keep_index = keep_index[:, n_post_nms]

        roi = roi[keep_index, :]

        return roi


