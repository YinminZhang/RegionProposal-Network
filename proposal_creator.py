import torch
from model.nms import gpu_nms
from model.bbox import loc2bbox

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
        score = score.squeeze(dim=0)[keep, 1]

        if n_pre_nms > 0:
            order = score.sort(0, descending=True)[1]
            n_pre_nms = min(n_pre_nms, roi.shape[0])
            roi = roi[order[:n_pre_nms], :]
            score = score[order[:n_pre_nms]]

        keep_index, _ = gpu_nms(torch.cat((roi, score.unsqueeze(dim=1)), dim=1), self.nms_threshold)
        keep_index = keep_index[:n_post_nms]

        roi = roi[keep_index, :]

        return roi