import torch
from losses import iou

def batch_gpu_nms(dets, thresh):
    """
    Implement NMS in gpu
    :param dets: (Batch, N, 5)bbox and score
    :param thresh: (1) the NMS algorithm thresh
    :return:
        keep_idx: list of keeping index
        keep:(N) 0-1 mask
    """
    scores = dets[:, 4]
    order = scores.sort(0, descending=True)

    batch_size = dets.shape[0].item()
    keep_idx = []
    keep = torch.zeros(batch_size)
    boxs = dets[:, :4]

    for idx in order[1]:
        if scores[idx] != 0:

            box_idx = boxs[idx]
            box_idx = torch.stack([box_idx]*batch_size, dim=0)
            _iou = iou(box_idx, boxs)

            scores = torch.where(_iou>thresh, torch.zeros_like(scores).cuda(), scores)

            keep_idx.append(idx)
            keep[idx] = 1

    return keep_idx, keep


def gpu_nms(dets, thresh, confidence=0.0):
    """
    Implement NMS in gpu
    :param dets: (N, 5)bbox and score
    :param thresh: (1) the NMS algorithm thresh
    :return:
        keep_idx: list of keeping index
        keep:(N) 0-1 mask
    """
    scores = dets[:, 4]
    order = scores.sort(0, descending=True)

    keep_idx = []
    keep = torch.zeros(scores.shape)
    boxs = dets[:, :4]

    scores_keep = torch.stack([order[0], order[1].float()], dim=1)
    scores_keep = scores_keep[scores_keep[:, 0]>confidence, :]
    boxs_keep = boxs[scores_keep[:, 1].long(), :]

    boxs_keep, scores_keep = delect_box(boxs_keep, scores_keep)

    while scores_keep.shape[0] > 0:
        idx = scores_keep[0, 1].long()
        box = boxs[idx]

        _iou = iou(torch.stack([box]*scores_keep.shape[0], dim=0), boxs_keep)

        scores_keep = scores_keep[_iou<thresh, :]
        boxs_keep = boxs_keep[_iou<thresh, :]

        keep_idx.append(idx)
        keep[idx] = 1

    return keep_idx, keep


def delect_box(bboxs, score):
    w = bboxs[:, 2] - bboxs[:, 0]
    h = bboxs[:, 3] - bboxs[:, 1]

    bboxs = bboxs[(w>0) & (h> 0), :]
    score = score[(w>0) & (h> 0), :]

    return bboxs, score