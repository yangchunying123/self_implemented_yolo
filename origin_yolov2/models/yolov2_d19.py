import torch
import torch.nn as nn
from .darknet19 import build_darknet19 as build_backbone
from .modules import Conv
from dataset.config import yolov2_d19_cfg as cfg



class YOLOv2D19(nn.Module):
    def __init__(self, device):
        super(YOLOv2D19, self).__init__()
        self.device = device
        self.num_anchors = len(cfg['anchor_size_voc'])
        self.stride =cfg['stride']
        self.backbone = build_backbone(pretrained=True)
        self.convsets_1 = nn.Sequential(
            Conv(1024, 1024, k=3, p=1),
            Conv(1024, 1024, k=3, p=1)
        ).cuda()
        # self.route_layer = Conv(512, 64, k=1).cuda()
        # self.reorg = reorg_layer(stride=2).cuda()
        # self.convsets_2 = Conv(1280, 1024, k=3, p=1).cuda()
        self.pred = nn.Conv2d(1024, self.num_anchors*(1 + 4 + cfg['class_num']), kernel_size=1).cuda()


    def forward(self, x):
        # backbone
        feats = self.backbone(x)
        # reorg layer
        p5 = self.convsets_1(feats['layer3'])
        # p4 = self.reorg(self.route_layer(feats['layer2']))
        # p5 = torch.cat([p4, p5], dim=1)
        # p5 = self.convsets_2(p5)
        pred = self.pred(p5)
        B, abC, H, W = pred.size()
        pred = pred.permute(0, 2, 3, 1).contiguous() # B, H, W, anchor_num * (5 + class)
        pred = pred.view(B, H, W, self.num_anchors, -1).contiguous()
        return pred
        
if __name__ == '__main__':
    bs = 16
    
    from dataset.config import yolov2_d19_cfg as cfg
    anchor_size = cfg['anchor_size_voc'] 
    device=torch.device('cuda')
    yolo = YOLOv2D19(device=device, anchor_size=anchor_size, input_size=320)
    for i in range(10, 19):
        tensor = torch.rand(bs, 3, 32 * i , 32 *i).to(device)
        prediction = yolo(tensor)
        print(prediction.size())
