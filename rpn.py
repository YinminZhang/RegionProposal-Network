import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.anchors import Anchors
from model.proposal_creator import ProposalCreator
from model.anchor_target_creator import AnchorTargetCreator


# TODO overwrite Reigon proposal network
class RPN(nn.Module):
    """ Reigon proposal network"""
    def __init__(self, in_channels, mid_channels,
                 scales = [8, 16, 32], ratios = [0.5, 1, 2],
                 stride = 16,
                 proposal_creator_params=dict()):
        super(RPN, self).__init__()

        # intial hyper-parameters
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.anchor_scales = scales
        self.anchor_ratios = ratios
        self.feat_stride = stride
        self.num_anchor = len(self.anchor_ratios) * len(self.anchor_scales)
        self.anchor = Anchors(pyramid_levels=[4])

        # intial network
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1, 1)
        # softmax bg/fg
        self.score_out = self.num_anchor * 2
        self.cls_score = nn.Conv2d(mid_channels, self.score_out, 1, 1, 0)
        # box offset
        self.loc_out = self.num_anchor * 4
        self.loc_pred = nn.Conv2d(mid_channels, self.loc_out, 1, 1, 0)

        # define proposal layer
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)

        # define anchor target layer
        self.anchor_target_layer = AnchorTargetCreator(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # intial network parameters
        _normal_init(self.conv1)
        _normal_init(self.cls_score)
        _normal_init(self.loc_pred)

        # intial loss
        self.rpn_cls_loss = 0
        self.rpn_box_loss = 0

    @staticmethod
    def reshape(feat, d):
        input_shape = feat.shape
        feat = feat.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1]*input_shape[2])/float(d)),
            input_shape[3]
        )
        return feat

    def forward(self, feat, img_size, scale=1):
        batch_size, channels, height, width = feat.shape
        anchors = self.anchor(np.ones(shape=(batch_size, 3, height * self.feat_stride,
                                             width * self.feat_stride)))[0]

        num_anchors = anchors.shape[0] // (height * width)

        conv = F.relu(self.conv1(feat))

        # location predict branch
        rpn_loc = self.loc_pred(conv)
        rpn_loc = rpn_loc.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)

        # classification prediction (background or foreground)
        rpn_scores = self.cls_score(conv)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
        rpn_softmax_scores = F.softmax(rpn_scores, dim=2)
        rpn_fg_scores = rpn_softmax_scores[:, 1].contiguous()

        # create rois
        rois = []
        roi_indices = []
        for i in range(batch_size):
            roi = self.proposal_layer(rpn_loc[i], rpn_scores[i],
                                      anchors, img_size, scale=scale)
            batch_index = i * torch.ones((len(roi),), dtype=torch.int32)
            rois.append(roi)
            roi_indices.append(batch_index)

        rois = torch.cat(rois, dim=0)
        roi_indices = torch.cat(roi_indices, dim=0)

        return rpn_loc, rpn_scores, rois, roi_indices, anchors

def _normal_init(m, mean=0, stddev=0.01, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
    else:
        nn.init.normal_(m.weight, mean, stddev)
        nn.init.zeros_(m.bias)

def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    for i in sorted(dim, reverse=True):
        loss_box = loss_box.sum(i)
    loss_box = loss_box.mean()
    return loss_box