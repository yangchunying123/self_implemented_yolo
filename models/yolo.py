import torch
import torch.nn as nn
from .darknet19 import build_darknet19 as build_backbone
from dataset.config import CLASS_NUM, yolov2_d19_cfg



class YOLO(nn.Module):
    def __init__(self, pretrained_backbone = False):
        super(YOLO, self).__init__()
        self.backbone = build_backbone(pretrained=pretrained_backbone)
        self.pred_head = nn.Sequential(nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(1024), 
                                    nn.LeakyReLU(0.1),
                                    nn.Conv2d(1024, len(yolov2_d19_cfg['anchor_size_voc'])*(5 + CLASS_NUM), kernel_size=1))

    def forward(self, x):
        '''
        x : [batch , 3, H, W]
        '''
        feature = self.backbone(x)      
        pred = self.pred_head(feature['layer3']) # output [batch, anchor * (5 + class), h, w]
        B, abC, h, w = pred.size()
        pred = pred.permute(0, 2, 3, 1).contiguous().view(B, h, w, len(yolov2_d19_cfg['anchor_size_voc']), -1)
        return pred


class YOLOLOSS(nn.Module):
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5):
        super(YOLOLOSS, self).__init__()
        self.mseloss = nn.MSELoss(reduction='sum')
        self.bceloss = nn.BCEWithLogitsLoss(reduction='sum')
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
    
    def forward(self, pred, gt):
        assert torch.all((gt[..., 4] == 0) | (gt[..., 4] == 1)), "Confidence labels must be 0 or 1."
        base = gt.size(-1)
        obj_mask = (gt[..., 4] == 1.0).unsqueeze(-1).expand_as(pred)
        noobj_mask = (gt[..., 4] == 0.0).unsqueeze(-1).expand_as(pred)
        gt_with_obj = gt[obj_mask].view(-1, base)
        gt_without_obj = gt[noobj_mask].view(-1, base)

        pred_with_obj = pred[obj_mask].view(-1, base)
        pred_without_obj = pred[noobj_mask].view(-1, base)
        
        xyloss = self.lambda_coord * self.mseloss(gt_with_obj[..., :2], pred_with_obj[..., :2])
        whloss = self.lambda_coord * self.mseloss(gt_with_obj[..., 2: 4], pred_with_obj[..., 2: 4])
        confloss = self.bceloss(pred_with_obj[..., 4], gt_with_obj[..., 4]) + self.lambda_noobj * self.bceloss(pred_without_obj[..., 4], gt_without_obj[..., 4]) 
        classloss = self.bceloss(pred_with_obj[..., 5:], gt_with_obj[..., 5:]) 
        loss = xyloss + whloss + confloss + classloss
        # print('xy loss {}, wh loss {}, conf loss {}, class loss {}'.format(xyloss, whloss, confloss, classloss))
        return loss / pred.size(0)
    



        
if __name__ == '__main__':
    model = YOLO(pretrained_backbone=True)
    tensor = torch.rand((1, 3, 416, 416))
        
    gt = torch.rand((1, 13, 13, 5, 25))
    gt[..., 4] = 0
    gt[..., 0, 4] = 1
    pred = model(tensor)
    print('pred',  pred.shape)
    print('gt', gt.shape)

    loffunc = YOLOLOSS()


    loss = loffunc(pred, gt)
    print(loss)