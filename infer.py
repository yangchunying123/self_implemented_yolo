import torch

def inference(weights, img_path):
    from PIL import Image
    from utils.tools import ImageTransfrom
    import cv2
    from dataset.config import CLASS

    img = Image.open(img_path).convert('RGB')
    transform_infer = ImageTransfrom(416)
    img = transform_infer(img)
    img = img.unsqueeze(0)

    from models.yolo import YOLO
    yolo = YOLO(pretrained_backbone=True)
    yolo.load_state_dict(torch.load(weights, weights_only=True))

    with torch.no_grad():
        preds = yolo(img)
    xyxycc = decode(preds)
    resized_image = cv2.resize(cv2.imread(img_path), (416, 416), interpolation=cv2.INTER_LANCZOS4)
    for item in xyxycc:
        x1, y1, x2, y2, _, classid = item
        x1, y1, x2, y2, classid = int(x1), int(y1), int(x2), int(y2), int(classid)
        cv2.rectangle(resized_image, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(resized_image, CLASS[classid], (x1, y1 -10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imwrite('haha.png', resized_image)





def decode(preds:torch.Tensor):
    from rich import print
    assert preds is not None, 'wtf?????????'
    from dataset.config import yolov2_d19_cfg, CONF_THRESHOLD, STRIDE, SUPRESSION
    from utils.tools import nms

    anchors = yolov2_d19_cfg['anchor_size_voc']
    anchors = torch.tensor(anchors)
    grid_size = preds.size(1)
    
    #([1, 13, 13, 5, 25])

    grid_y, grid_x = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size))
    grid_x = grid_x.view(grid_size, grid_size, 1).float()
    grid_y = grid_y.view(grid_size, grid_size, 1).float()
    pred = preds[0]


    x = torch.sigmoid(pred[..., 0]) 

    x = (x + grid_x) * STRIDE
    y = (torch.sigmoid(pred[..., 1]) + grid_y) * STRIDE
    w = (torch.exp(pred[..., 2]) * anchors[:, 0]) * STRIDE
    h = (torch.exp(pred[..., 3]) * anchors[:, 1]) * STRIDE

    x1 , y1, x2, y2 = x - 0.5 * w, y - 0.5 * h, x + 0.5 * w, y + 0.5 * h
    conf = torch.sigmoid(pred[..., 4])
    class_id = torch.argmax(torch.sigmoid(pred[..., 5:]), dim=-1)
    xycc = torch.stack([x1, y1, x2, y2, conf, class_id], dim=-1)
    mask = xycc[..., 4] > CONF_THRESHOLD
    over_conf = xycc[mask.unsqueeze(-1).expand_as(xycc)].view(-1, 6)
    results = nms(over_conf, threshold = SUPRESSION)
    return results

    




if __name__ == '__main__':
    weights = '/home/asher/codes/python/self_yolo_imple_serires/deepseekV2/old_weights.pth'
    img = '/home/asher/codes/python/self_yolo_imple_serires/tmp/V1/zhupipi.jpg'
    # img = '/home/asher/codes/python/self_yolo_imple_serires/DATASET/VOC/2007/JPEGImages/000012.jpg'
    inference(weights, img)
