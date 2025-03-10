import torch
import torch.nn as nn
from models.basic import Conv
from rich import print
from torchvision.models.resnet import resnet18, ResNet18_Weights

# YOLO
class YOLOV1(nn.Module):
    def __init__(self, B=2, C=20):
        super(YOLOV1, self).__init__()
        self.C = C
        self.B = B             
        features = resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(features.children())[:-2])
        feat_dim = 512
        # detection head
        self.convsets = nn.Sequential(
            Conv(feat_dim, feat_dim//2, k=1),
            Conv(feat_dim//2, feat_dim, k=3, p=1),
            Conv(feat_dim, feat_dim//2, k=1),
            Conv(feat_dim//2, feat_dim, k=3, p=1)
        )
        # pred
        self.pred = nn.Conv2d(feat_dim, 5*B + C, 1)
        nn.init.constant_(self.pred.bias, 0)  # 其他通道初始化为0
        # 仅对置信度部分的偏置进行特殊初始化
        with torch.no_grad():
            confidence_bias = -torch.log(torch.tensor(99.0))
            for b in range(self.B):
                self.pred.bias[:5*b].fill_(confidence_bias)

    def forward(self, x):
        # backbone主干网络
        feat = self.backbone(x)
        feat = self.convsets(feat)
        pred = self.pred(feat)
        pred = pred.permute(0, 2, 3, 1).contiguous()
        pred[..., : 5 * self.B] = torch.sigmoid(pred[..., : 5 * self.B])
        pred[..., 5 * self.B : ] = torch.softmax(pred[..., 5 * self.B: ], dim=-1)
        return pred
         
if __name__ == '__main__':
    # t = torch.rand(1, 3, 416, 416)
    model = YOLOV1()
    # output = model(t)
    # print(output[0, 0, 0 , :])
    # print(torch.sum(output[0,0,0, 10 : ]))
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
