import torch
import random
import os
from torchvision import transforms
from dataset.config import yolov2_d19_cfg, STRIDE, CLASS_NUM

class ImageTransfrom:
    def __init__(self, image_size):
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def __call__(self, raw_img):
        return self.transform(raw_img)

def iou(bbox1, bbox2, single_compare = False):
    if isinstance(bbox1, list) and isinstance(bbox2, list):
        tensor1 = torch.zeros((len(bbox1), 4))
        tensor2 = torch.zeros((len(bbox2), 4))
        if single_compare:
            bbox1 = torch.tensor(bbox1[:4]).unsqueeze(0)
            bbox2 = torch.tensor(bbox2[:4]).unsqueeze(0)
            result = __iou__(torch.tensor(bbox1[:4]), torch.tensor(bbox2[:4]))
            return result.item()
        else:
            for i in range(len(bbox1)):
                tensor1[i, : ] = torch.tensor([0, 0, bbox1[i][0], bbox1[i][1]]) 
            for i in range(len(bbox2)):
                tensor2[i, : ] = torch.tensor([0, 0, bbox2[i][0], bbox2[i][1]]) 
        return __iou__(tensor1, tensor2)


    elif isinstance(bbox1, torch.Tensor) and isinstance(bbox2, torch.Tensor):
        shape1, shape2 = bbox1.shape, bbox2.shape 
        assert shape1[-1] == shape2[-1], 'Last demension of boxes must be the same, Box1 shape {}, and Box2 shape {}'.format(shape1, shape2)
        assert shape1[-1] == 2 or shape1[-1] == 4, 'Last demension must be 2 or 4, but got {}'.format(shape1[-1])
        if shape1[-1] == 2:
            bbox1 = torch.cat([torch.zeros((shape1[0], 2)), bbox1], dim=-1)
            bbox2 = torch.cat([torch.zeros((shape2[0], 2)), bbox2], dim=-1)
        return __iou__(bbox1, bbox2)
      

def __iou__(bbox1:torch.Tensor, bbox2: torch.Tensor):
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

def detection_collate(batch):
    scale_ops = [STRIDE * i for i in range(yolov2_d19_cfg['random_size_range'][0], yolov2_d19_cfg['random_size_range'][1])]
    anchors = yolov2_d19_cfg['anchor_size_voc']
    imgs = []
    labels = []

    scale = random.choice(scale_ops)
    img_trans = ImageTransfrom(scale)
    for img, label in batch:
        grid_size = int(scale / STRIDE)
        
        image_tensor = img_trans(img)
        label_tensor = torch.zeros((grid_size, grid_size, len(anchors), 5 + CLASS_NUM))
        for bbox in label:
            x1, y1, x2, y2 , classid = bbox
            centerx = (x1 + x2)/2
            centery = (y1 + y2)/2
            width, height = x2 - x1, y2 - y1
            grid_x = int(centerx * grid_size)
            grid_y = int(centery * grid_size)
            tx = centerx * grid_size - grid_x
            ty = centery * grid_size - grid_y

            grid_width = width * grid_size
            grid_height = height * grid_size
            iou_result = iou([[grid_width, grid_height]],  anchors)
            _, best_anchor = torch.max(iou_result, dim=-1)
            tw = torch.log(torch.tensor(grid_width/anchors[best_anchor][0] + 1e-10))
            th = torch.log(torch.tensor(grid_height/anchors[best_anchor][1] + 1e-10))
            label_tensor[grid_y, grid_x, best_anchor, :4] = torch.tensor([tx, ty, tw, th])
            label_tensor[grid_y, grid_x, best_anchor, 4] = 1.0
            label_tensor[grid_y, grid_x, best_anchor, 5 + classid] = 1.0
        imgs.append(image_tensor)
        labels.append(label_tensor)
    return torch.stack(imgs, 0), torch.stack(labels, 0)

                
def nms(candicates:torch.Tensor, threshold=0.7):
    sorted_tensor, sorted_indices = torch.sort(candicates[..., 4], descending=True)
    sorted = candicates[sorted_indices]
    xylist = sorted.tolist()

    final = []
    while xylist:
        base = xylist.pop(0)
        final.append(base)
        xylist = [item for item in xylist if iou(base, item, single_compare=True) < threshold]
    return final


if __name__ == '__main__':
    N = 3
    M = 4
    x1y1 = torch.rand(N, M)
    print(x1y1)
    max_v, max_idx = torch.max(x1y1, dim=0)
    print(max_v)
    print(max_idx)

    # print(x1y1)
    # # ele1 = torch.cat([x1y1, x1y1 + torch.rand(N, 2)], dim=-1)
    # # ele1[:, 2:][ele1[:, 2:] > 1] = 1.0
    # # x1y1 = torch.rand(M, 2)
    # # ele2 = torch.cat([x1y1, x1y1 + torch.rand(M, 2)], dim=-1)
    # # ele2[:, 2:][ele2[:, 2:] > 1] = 1.0
    # # print(ele1, '\n', ele2)
    # # result = __iou__(ele1, ele2)
    # # print(result.shape, '\n', result)
    # result = iou(x1y1, torch.rand(M, 2))
    # print(result.shape, '\n', result)
    # print('===== test for list ======')
    # list1 = []
    # list2 = []
    # import random
    # for i in range(N):
    #     list1.append([random.random(), random.random()])
    # for i in range(M):
    #     list2.append([random.random(), random.random()])
    # result = iou(list1, list2)
    # print(result.shape, result)
    