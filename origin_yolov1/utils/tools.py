import torch
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

def create_grid(S, batch = None):
    grid_y, grid_x = torch.meshgrid([torch.arange(S), torch.arange(S)])
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
    grid_xy = grid_xy.view(S, S, 2)
    if batch is not None:
        grid_xy = grid_xy.unsqueeze(0).expand([batch, S, S, 2])
    return grid_xy

def visualize_encoded_tensor(img, encoded_tensor):
    orig_h, orig_w , _ = img.shape
    N = encoded_tensor.size(-1)
    S = encoded_tensor.size(1)
    B = int((N - len(VOC_CLASS_BGR)) // 5)

    grid = create_grid(S)
    for b in range(B):
        encoded_tensor[..., 5 * b: 5 * b + 2] = (encoded_tensor[..., 5 * b: 5 * b + 2] + grid ) / S

    mask = (encoded_tensor[..., 4] == 1).unsqueeze(-1).expand_as(encoded_tensor)
    obj_lines = encoded_tensor[mask].view(-1, N)
    for i in range(obj_lines.size(0)):
        obj = obj_lines[i]
        x, y, w , h = obj[0].item(), obj[1].item(), obj[2].item(), obj[3].item()        
        _, index = torch.max(obj[5*B:], dim=0)
        classstr, color  = list(VOC_CLASS_BGR.items())[index]
        x1, x2, y1, y2 = x - 0.5 * w, x + 0.5 * w, y - 0.5 * h, y + 0.5 * h
        x1, y1, x2, y2 = int(x1 * orig_w), int(y1 * orig_h), int(x2 * orig_w), int(y2 * orig_h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, classstr, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 255, 255), thickness=1, lineType=8)
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



