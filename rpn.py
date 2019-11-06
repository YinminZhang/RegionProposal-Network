import torch
import torch.nn as nn
import torch.nn.functional as F
from proposal_creator import ProposalCreator
from anchor_target_creator import AnchorTargetCreator


# TODO overwrite Reigon proposal network
class RPN(nn.Module):
    """ Reigon proposal network"""
    def __index__(self,
                  in_channels,
                  mid_channels,
                  scales = [8, 16, 32],
                  ratios = [0.5, 1, 2],
                  stride = 16,
                  ):
        super(RPN, self).__init__()

        # intial hyper-parameters
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.anchor_scales = scales
        self.anchor_ratios = ratios
        self.feat_stride = stride
        self.n_anchor = len(self.anchor_ratios) * len(self.anchor_scales)
        # intial network
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1, 1)
        # softmax bg/fg
        self.score_out = self.n_anchor * 2
        self.cls_score = nn.Conv2d(mid_channels, self.score_out, 1, 1, 0)
        # box offset
        self.bbox_out = self.n_anchor * 4
        self.bbox_pred = nn.Conv2d(mid_channels, self.bbox_out, 1, 1, 0)

        # define proposal layer
        self.RPN_proposal = ProposalCreator(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # define anchor target layer
        self.RPN_anchor_target = AnchorTargetCreator(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # intial network parameters
        _normal_init(self.conv1)
        _normal_init(self.cls_score)
        _normal_init(self.bbox_pred)

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

    def forward(self, base_feature, im_info, gt_boxes, num_boxes):

        batch_size = base_feature.shape[0]

        # return feature map after conv + relu layer
        rpn_conv1 = F.relu(self.conv1(base_feature), inplace=True)

        # get rpn cls score
        rpn_cls_score = self.cls_score(rpn_conv1)
        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.score_out)

        # get rpn offset to the anchor boxes
        rpn_bbox_pred = self.bbox_pred(rpn_conv1)

        cfg_key = 'train' if self.training else 'test'

        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data,
                                  im_info, cfg_key))

        self.rpn_cls_loss = 0
        self.rpn_box_loss = 0

        # See https://github.com/jwyang/faster-rcnn.pytorch/blob/master/lib/model/rpn/anchor_target_layer.py for more details.
        if self.training:

            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, num_boxes))

            # compute cls loss
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            rpn_label = rpn_data[0].view(batch_size, -1)

            rpn_keep = torch.tensor(rpn_label.view(-1).ne(-1).nonzero().view(-1))
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1, 2), 0, rpn_keep)
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = torch.tensor(rpn_label.long())
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
            fg_cnt = torch.sum(rpn_label.data.ne(0))

            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]

            # compute bbox regression loss
            rpn_bbox_inside_weights = torch.tensor(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = torch.tensor(rpn_bbox_outside_weights)
            rpn_bbox_targets = torch.tensor(rpn_bbox_targets)

            self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                rpn_bbox_outside_weights, sigma=3, dim=[1, 2, 3])

        return rois, self.rpn_loss_cls, self.rpn_loss_box



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