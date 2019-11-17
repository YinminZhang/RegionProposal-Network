import torch
import numpy as np
from utils.losses import iou
from model.bbox import bbox2loc


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
        anchors[:, 1] = torch.clamp_min(anchors[:, 1], 0)
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
        positive_index = (targets == 1).nonzero().view(-1)

        if len(positive_index) > num_positive:
            # disable_index =  np.random.choice(positive_index, size=(len(positive_index) - num_positive), replace=False)
            # disable_index = torch.from_numpy(disable_index).cuda()
            disable_index = torch.multinomial(positive_index.float(), num_samples=(positive_index.shape[0] - num_positive), replacement=False).long()
            targets[disable_index] = -1

        # subsample negative labels if too many labels are negative
        num_negative = self.n_sample - (targets == 1).sum()
        neg_index = (targets == 0).nonzero().view(-1)
        if len(neg_index) > num_negative:
            # disable_index = np.random.choice(neg_index, size=(len(neg_index) - num_negative), replace=False)
            # disable_index = torch.from_numpy(disable_index).cuda()
            disable_index = torch.multinomial(neg_index.float(), num_samples=(neg_index.shape[0] - num_negative), replacement=False).long()
            targets[disable_index] = -1

        loc = bbox2loc(anchors, bbox[IoU_argmax])

        return loc, targets
