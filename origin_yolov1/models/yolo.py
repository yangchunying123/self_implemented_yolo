import torch
import torch.nn as nn
from models.basic import Conv
from models.backbone.resnet import build_resnet
from rich import print

# YOLO
class YOLOV1(nn.Module):
    def __init__(self, B = 2, C=20, load_backbone_weight=True):
        super(YOLOV1, self).__init__()
        self.C = C
        self.B = B             
        # backbone: resnet18
        self.backbone, feat_dim = build_resnet('resnet18', pretrained=load_backbone_weight)
        # detection head
        self.convsets = nn.Sequential(
            Conv(feat_dim, feat_dim//2, k=1),
            Conv(feat_dim//2, feat_dim, k=3, p=1),
            Conv(feat_dim, feat_dim//2, k=1),
            Conv(feat_dim//2, feat_dim, k=3, p=1)
        )
        # pred
        self.pred = nn.Conv2d(feat_dim, 5*B + C, 1)

    def forward(self, x):
        # backbone主干网络
        feat = self.backbone(x)
        feat = self.convsets(feat)
        pred = self.pred(feat)
        pred = pred.permute(0, 2, 3, 1).contiguous()
        return pred
    
def load(path):
    pretrained_state = torch.load(path)
    myyolo = YOLOV1(load_backbone_weight=False)
    state_dict_yolo = myyolo.state_dict()
    
    # 记录成功加载的键和缺失的键
    matched_keys = []
    missing_keys = []
    
    # 遍历模型 B 的键，检查是否在模型 A 中存在且形状匹配
    for key in state_dict_yolo:
        if key in pretrained_state:
            # 形状相同则加载权重
            if pretrained_state[key].shape == state_dict_yolo[key].shape:
                state_dict_yolo[key] = pretrained_state[key]
                matched_keys.append(key)
            else:
                missing_keys.append(f"{key} (shape mismatch)")
        else:
            missing_keys.append(f"{key} (key not found)")
    
    # 将更新后的权重加载到模型 B
    myyolo.load_state_dict(state_dict_yolo)
    return myyolo, matched_keys, missing_keys
         
if __name__ == '__main__':
    # yolo = YOLOV1().cuda()
    # from datas.voc import VOCDataset, ImageTransfrom
    # from torch.utils.data import DataLoader
    # image_dir = '/home/asher/codes/python/self_yolo_imple_serires/DATASET/VOC/2007/JPEGImages'
    # train_label = '/home/asher/codes/python/self_yolo_imple_serires/DATASET/train.txt'
    # val_label = '/home/asher/codes/python/self_yolo_imple_serires/DATASET/val.txt'

    # image_size = 416
    # train_dataset = VOCDataset(image_dir, train_label, image_trans=ImageTransfrom(image_size), grid_size=int(image_size/32))
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)

    # for i, (imgs, targets) in enumerate(train_loader):
    #     imgs, targets = imgs.cuda(), targets.cuda()
    #     print(i, imgs.shape)
    #     preds = yolo(imgs)
    #     print(preds.shape)
    
    # model, match, miss = load('/home/asher/Downloads/yolo_69.6.pth')
    # model = model.cuda()
    # t = torch.rand(3, 3, 416, 416)
    # print(model(t).shape)

    from torchvision.models.resnet import resnet50, ResNet50_Weights
    backbone = resnet50(weights = ResNet50_Weights.IMAGENET1K_V1)
    features = nn.Sequential(*list(backbone.children())[:-2])

    t = torch.rand(1, 3, 416, 416)
    out = features(t)
    print(out.shape)
