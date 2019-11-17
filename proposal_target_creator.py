import torch
import numpy as np
from model.bbox import bbox2loc
from utils.losses import iou

class ProposalTargetCreator:
    def __init__(self, num_sample=128, positive_ratio=0.25,
                 positive_iou_threshold=0.5,
                 negative_iou_threshold_high=0.5,
                 negative_iou_threshold_low=0.0):
        self.num_sample = num_sample
        self.positive_ratio = positive_ratio
        self.positive_iou_threshold = positive_iou_threshold
        self.negative_iou_threshold_high = negative_iou_threshold_high
        self.negative_iou_threshold_low = negative_iou_threshold_low

    def __call__(self, roi, bbox, label,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):

        num_bbox = bbox.shape[0]

        roi = torch.cat((roi, bbox), dim=0)

        positive_roi_per_image = torch.floor(torch.tensor(self.num_sample * self.positive_ratio))
        IoU = iou(roi, bbox)

        max_iou, gt_assignment = torch.max(IoU, dim=1)
        gt_roi_label = label[gt_assignment] + 1
        bbox_assignment = bbox[gt_assignment, :]

        positive_index = max_iou >= self.positive_iou_threshold
        positive_roi_per_image = int(min(positive_roi_per_image, positive_index.sum()))
        if positive_index.sum() > 0:
            # positive_index = np.random.choice(positive_index.nonzero().view(-1).cpu().numpy(), size=positive_roi_per_image, replace=False)
            # positive_index = torch.from_numpy(positive_index).cuda()
            positive_index = torch.multinomial(positive_index.nonzero().view(-1).float(), num_samples=positive_roi_per_image, replacement=False).long()

        print(IoU)
        print(max_iou)
        print((max_iou < self.negative_iou_threshold_high), (max_iou >= self.negative_iou_threshold_low))
        negative_index = (max_iou < self.negative_iou_threshold_high) & (max_iou >= self.negative_iou_threshold_low)
        negative_roi_per_image = self.num_sample - positive_roi_per_image
        negative_roi_per_image = int(min(negative_roi_per_image, negative_index.sum()-1))
        negative_index = negative_index.nonzero().view(-1)

        if negative_index.shape[0] > 1:
            print(max_iou.shape, positive_index.shape, negative_index.shape, negative_roi_per_image, negative_index)
            # negative_index = np.random.choice(negative_index.nonzero().view(-1).cpu().numpy(), size=negative_roi_per_image, replace=False)
            # negative_index = torch.from_numpy(negative_index).cuda()
            negative_index = torch.multinomial(negative_index.float().cpu(), num_samples=negative_roi_per_image, replacement=False).long().cuda()

        print(positive_index.shape, negative_index.shape)
        keep = torch.cat((positive_index, negative_index)).long()
        gt_roi_label = gt_roi_label[keep]
        gt_roi_label[-negative_roi_per_image:] = 0
        sample_roi = roi[keep, :]

        gt_roi_loc = bbox2loc(sample_roi, bbox_assignment[keep, :])
        gt_roi_loc = ((gt_roi_loc - torch.tensor(loc_normalize_mean).float().cuda()
                       ) / torch.tensor(loc_normalize_std).float().cuda())

        return sample_roi, gt_roi_loc, gt_roi_label



