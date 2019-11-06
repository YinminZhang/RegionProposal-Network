import numpy as np
import torch
def _whctrs(acnhors):
    h = acnhors[:, 2] - acnhors[:, 0] + 1.0
    w = acnhors[:, 3] - acnhors[:, 1] + 1.0
    y = acnhors[:, 0] + 0.5 * h
    x = acnhors[:, 1] + 0.5 * w
    return h, w, y, x


def loc2bbox(src_bbox, loc):
    src_bbox = src_bbox.astype(src_bbox.dtyep, copy=False)

    src_height, src_weight, src_ctr_y, src_ctr_x = _whctrs(src_bbox)

    dy = loc[:, 0::4]
    dx = loc[:, 1::4]
    dh = loc[:, 2::4]
    dw = loc[:, 3::4]

    ctr_y = dy * src_height.unsqueeze(2) + src_ctr_y.unsqueeze(2)
    ctr_x = dx * src_weight.unsqueeze(2) + src_ctr_x.unsqueeze(2)

    h = torch.exp(dh) * src_height.unsqueeze(2)
    w = torch.exp(dw) * src_weight.unsqueeze(2)

    bbox = loc.clone()
    bbox[:, 0::4] = ctr_y - 0.5*h
    bbox[:, 1::4] = ctr_x - 0.5*w
    bbox[:, 2::4] = ctr_y + 0.5*h
    bbox[:, 3::4] = ctr_x + 0.5*w

    return bbox

def bbox2loc(src_bbox, bbox):
    h, w, ctr_y, ctr_x = _whctrs(src_bbox)

    base_height, base_weight, base_ctr_y, base_ctr_x = _whctrs(bbox)

    dy = (base_ctr_y - ctr_y) / h
    dx = (base_ctr_x - ctr_x) / w
    dw = torch.log(base_weight / w)
    dh = torch.log(base_height / h)

    loc = torch.stack((dy, dx, dw, dh), 1)

    return loc

def np_iou(bbox_a, bbox_b):
    tl = np.maximum(bbox_a[, None, :2], bbox_b[:, :2])
    br = np.minimum(bbox_a[, None, 2:], bbox_b[:, 2:])

    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)

def iou(bbox_a, bbox_b):
    # TODO compute IoU using torch

    tl = torch.max(bbox_a[:,:2], bbox_b[:, :2])
    br = torch.min(bbox_a[:,2:], bbox_b[:, 2:])

    area_i = torch.prod(br - tl, dim=1) * (tl < br).all(dim=1).float()
    area_a = torch.prod(bbox_a[:, 2:] - bbox_a[:, :2], dim=1)
    area_b = torch.prod(bbox_b[:, 2:] - bbox_b[:, :2], dim=1)
    return area_i / (area_a + area_b - area_i)

# TODO add clip boxes
def clip_boxes(boxes, im_shape, batch_size):

    x = im_shape[:, 1] - 1
    y = im_shape[:, 0] - 1

    boxes[:, :, 0].clamp_(0, x)
    boxes[:, :, 1].clamp_(0, y)
    boxes[:, :, 2].clamp_(0, x)
    boxes[:, :, 3].clamp_(0, y)

    return boxes

# TODO add bbox overlaps
def bbox_overlaps(anchors, gt_boxes):
    """
       Parameters
       ----------
       boxes: (N, 4) ndarray of float
       gt_boxes: (K, 4) ndarray of float
       Returns
       -------
       overlaps: (N, K) ndarray of overlap between boxes and query_boxes
       """

    N = anchors.shape[0]
    K = gt_boxes.shape[0]

    anchors = torch.stack([anchors] * K, dim=1).view(-1, 4)
    gt_boxes = torch.stack([gt_boxes] * N, dim=0).view(-1, 4)

    overlaps = iou(anchors, gt_boxes)
    overlaps = overlaps.view(N, K)

    return overlaps

# TODO add batch size to generate IoU table (B * N * K)
def batch_bbox_overlaps(anchors, gt_boxes):
    """
           Parameters
           ----------
           boxes: (N, 4) ndarray of float
           gt_boxes: (B, K, 4) ndarray of float
           Returns
           -------
           overlaps: (B, N, K) ndarray of overlap between boxes and query_boxes
           """
    N = anchors.shape[0]
    B, K, _ = gt_boxes.shape

    anchors = torch.stack([anchors] * B * K, dim=1).view(-1, 4)
    gt_boxes = torch.stack([gt_boxes] * N, dim=0).view(-1, 4)

    overlaps = iou(anchors, gt_boxes)
    overlaps = overlaps.view(B, N, K)

    return overlaps