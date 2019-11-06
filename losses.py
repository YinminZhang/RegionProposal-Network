import torch

def iou(bbox_a, bbox_b):

    if bbox_a.shape[0] != bbox_b.shape[0]:
        tl = torch.max(bbox_a[:, :2].unsqueeze(dim=1), bbox_b[:, :2])
        br = torch.min(bbox_a[:, 2:].unsqueeze(dim=1), bbox_b[:, 2:])
        area_i = torch.prod(br - tl, dim=2) * (tl <= br).all(dim=2).float()
        area_a = torch.prod(bbox_a[:, 2:] - bbox_a[:, :2], dim=1)
        area_b = torch.prod(bbox_b[:, 2:] - bbox_b[:, :2], dim=1)
        area_u = torch.clamp(area_a.unsqueeze(dim=1) + area_b - area_i, min=1e-8)
    else:
        tl = torch.max(bbox_a[:,:2], bbox_b[:, :2])
        br = torch.min(bbox_a[:,2:], bbox_b[:, 2:])
        area_i = torch.prod(br - tl, dim=1) * (tl <= br).all(dim=1).float()
        area_a = torch.prod(bbox_a[:, 2:] - bbox_a[:, :2], dim=1)
        area_b = torch.prod(bbox_b[:, 2:] - bbox_b[:, :2], dim=1)
        area_u = torch.clamp(area_a + area_b - area_i, min=1e-8)

    return area_i / area_u