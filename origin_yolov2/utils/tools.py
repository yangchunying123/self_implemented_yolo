import torch
from dataset.config import yolov2_d19_cfg
from math import ceil


def compute_iou(bbox1, bbox2):
    N = bbox1.size(0)
    M = bbox2.size(0)
    lt = torch.max(
        bbox1[:, :2].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
        bbox2[:, :2].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
    )
    rb = torch.min(
        bbox1[:, 2:].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
        bbox2[:, 2:].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
    )
    wh = rb - lt   # width and height of the intersection, [N, M, 2]
    wh[wh < 0] = 0 # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1] # [N, M]
    area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1]) # [N, ]
    area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1]) # [M, ]
    area1 = area1.unsqueeze(1).expand_as(inter) # [N, ] -> [N, 1] -> [N, M]
    area2 = area2.unsqueeze(0).expand_as(inter) # [M, ] -> [1, M] -> [N, M]
    union = area1 + area2 - inter # [N, M]
    iou = inter / union           # [N, M]
    return iou

def compute_iou_centerxywh_format(bbox1, bbox2, max):
    b1 = torch.zeros_like(bbox1)
    b2 = torch.zeros_like(bbox2)
    b1[:, 0 : 2] = bbox1[:, 0 : 2] - 0.5 * bbox1[:, 2 : 4]
    b1[:, 2 : 4] = bbox1[:, 0 : 2] + 0.5 * bbox1[:, 2 : 4]
    b1[b1 < 0] = 0
    b1[b1 > max] = max
    b2[:, 0 : 2] = bbox2[:, 0 : 2] - 0.5 * bbox2[:, 2 : 4]
    b2[:, 2 : 4] = bbox2[:, 0 : 2] + 0.5 * bbox2[:, 2 : 4]
    b1[b1 < 0] = 0
    b1[b1 > max] = max
    return compute_iou(b1, b2)


def gt_creator(grid_num, label_lists):
    batch_size = len(label_lists)
    anchors = yolov2_d19_cfg['anchor_size_voc']
    gt_tensor = torch.zeros([batch_size, grid_num, grid_num, len(anchors), 5 + yolov2_d19_cfg['class_num']])

    for batch_index in range(batch_size):
        for box_class in label_lists[batch_index]:
            x, y, w, h, classid = box_class
            x, y, w, h = grid_num * x, grid_num * y, grid_num * w, grid_num * h
            i = ceil(x) - 1
            j = ceil(y) - 1
            cx = x - i
            cy = y - j
            box_tensor = torch.tensor([x, y, w, h], dtype=torch.float32).unsqueeze(0)
            anchor_tensor = torch.tensor(anchors, dtype=torch.float32) # [5, 2]
            anchor_centers = torch.tensor([i + 0.5, j + 0.5]).expand([len(anchors), 2]) # [5, 2]
            anchor_tensor = torch.cat([anchor_centers, anchor_tensor], dim= -1)
            iou = compute_iou_centerxywh_format(box_tensor, anchor_tensor, grid_num)
            targetiou, index = torch.max(iou, dim=1)
            best_anchor = anchors[index.item()]
            gt_tensor[batch_index, j, i, index.item(), : 5] = torch.tensor([cx, cy, w/best_anchor[0], h/best_anchor[1], targetiou.item()], dtype=torch.float32)
            gt_tensor[batch_index, j, i, index.item(), 5 + classid] = 1.0
    return gt_tensor

test_label = [[[0.45, 0.48, 0.2, 0.2, 1]]]
gt = gt_creator(2, test_label)
print(gt)