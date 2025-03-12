import torch
from dataset.config import yolov2_d19_cfg
from math import ceil
import cv2

VOC_CLASS_BGR = {
    'aeroplane': (128, 0, 0),
    'bicycle': (0, 128, 0),
    'bird': (128, 128, 0),
    'boat': (0, 0, 128),
    'bottle': (128, 0, 128),
    'bus': (0, 128, 128),
    'car': (128, 128, 128),
    'cat': (64, 0, 0),
    'chair': (192, 0, 0),
    'cow': (64, 128, 0),
    'diningtable': (192, 128, 0),
    'dog': (64, 0, 128),
    'horse': (192, 0, 128),
    'motorbike': (64, 128, 128),
    'person': (192, 128, 128),
    'pottedplant': (0, 64, 0),
    'sheep': (128, 64, 0),
    'sofa': (0, 192, 0),
    'train': (128, 192, 0),
    'tvmonitor': (0, 64, 128)
}

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

def batch_gt_tensor_creator(grid_num, batch_labels):
    gts = []
    for label in batch_labels:
        gts.append(gt_creator(grid_num, label))
    return torch.stack(gts, dim=0)

def batch_image_tensor_creator(grid_num, batch_img_tensors):
    return torch.nn.functional.interpolate(batch_img_tensors, size=grid_num * yolov2_d19_cfg['stride'], mode='bilinear', align_corners=False)

def gt_creator(grid_num, single_image_label_lists):
    anchors = yolov2_d19_cfg['anchor_size_voc']
    gt_tensor = torch.zeros([grid_num, grid_num, len(anchors), 6])

    for box_class in single_image_label_lists:
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
        gt_tensor[j, i, index.item(), : 5] = torch.tensor([cx, cy, w/best_anchor[0], h/best_anchor[1], targetiou.item()], dtype=torch.float32)
        gt_tensor[j, i, index.item(), 5] = classid
    return gt_tensor

def visualize_gt_tensor(image_path, gt_tensor:torch.Tensor):
    image = cv2.imread(image_path)
    orig_h, orig_w , _ = image.shape
    anchors = yolov2_d19_cfg['anchor_size_voc']
    if len(gt_tensor.shape) == 5:
        gt_tensor = gt_tensor.squeeze(0)
    H, W, anchor, last = gt_tensor.shape
    for j in range(H):
        for i in range(W):
            for anch in range(anchor):
                if gt_tensor[j, i, anch, 4] > 0:
                    xoff, yoff, widthoff, heightoff, conf, classid = gt_tensor[j, i , anch, : ]  
                    x, y, w, h, = xoff + i, yoff + j , widthoff * anchors[anch][0], heightoff * anchors[anch][1] 
                    x1, x2, y1, y2 = x - 0.5 * w, x + 0.5 * w, y - 0.5 * h, y + 0.5 * h
                    x1, x2, y1, y2 = int(x1 * orig_w / W), int(x2 * orig_w / W), int(y1 * orig_h / H), int(y2 * orig_h / H)
                    classstr, color  = list(VOC_CLASS_BGR.items())[int(classid.item())]
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(image, classstr, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 255, 255), thickness=1, lineType=8)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def yolo_collefn_in(batch):
    targets = []
    imgs = []
    for img, labels, paths in batch:
        imgs.append(img)
        targets.append(labels)
    return torch.stack(imgs, 0), targets