import torch
import torch.nn as nn
from rich import print
from dataset.config import yolov2_d19_cfg as cfg
class Loss(nn.Module):
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5):
        super(Loss, self).__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def create_priors(self, grid_num, anchors): # return [gridnum, gridnum, anchornum, 2], [gridnum, gridnum, anchornum, 2]
        grid_y, grid_x = torch.meshgrid([torch.arange(grid_num), torch.arange(grid_num)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        return grid_xy.view(grid_num, grid_num, -1, 2), anchors.repeat(1, 1, 1, 1)
    
    def transform_txtytwth2xywh(self, prediction_tensor:torch.Tensor):
        grid_num, batch_size = prediction_tensor.size(1), prediction_tensor.size(0)
        grid_prior, anchors = self.create_priors(grid_num, cfg['anchor_size_voc'])
        grid_prior, anchors = grid_prior.unsqueeze(0), anchors.unsqueeze(0)
        
        pre_tensor = prediction_tensor.clone()
        pre_tensor[..., : 2] = torch.sigmoid(pre_tensor[..., : 2]) + grid_prior
        pre_tensor[..., 2: 4] = torch.exp(pre_tensor[..., 2 : 4]) * anchors
        pre_tensor[..., 2: 4] = torch.clamp(pre_tensor[..., 2: 4], 0, grid_num)
        pre_tensor[..., 4] = torch.sigmoid(pre_tensor[..., 4])
        return pre_tensor


    def forward(self, prediction_tensor, target_tensor):
        """ Compute loss for YOLO training.
        Args:
            pred_tensor: (Tensor) predictions, sized [n_batch, S, S, anchor, 5 + class_num]).
            target_tensor: (Tensor) targets, sized [n_batch, S, S, anchor, 5 + 1], 5 : [offsetx, offsety, w, h, conf] + classid.
        Returns:
            (Tensor): loss, sized [1, ].
        """
        pred_tensor = self.transform_txtytwth2xywh(prediction_tensor)
        batch_size, N =  pred_tensor.size(0), pred_tensor.size(-1)
        obj_mask = (target_tensor[..., 4] > 0).unsqueeze(-1)
        no_obj_mask = (target_tensor[..., 4] == 0).unsqueeze(-1)
        
        obj_gt = target_tensor[obj_mask.expand_as(target_tensor)].view(-1, 6)
        obj_pre = pred_tensor[obj_mask.expand_as(pred_tensor)].view(-1, N)
        noobj_pre = pred_tensor[no_obj_mask.expand_as(pred_tensor)].view(-1, N)

        cls_loss_function = nn.CrossEntropyLoss(reduction='sum')
        mse_loss_function = nn.MSELoss(reduction='sum')

        wh_loss = self.lambda_coord * mse_loss_function(obj_pre[:, 2:4], obj_gt[:, 2:4])
        xy_loss = self.lambda_coord * mse_loss_function(obj_pre[:, : 2], obj_gt[:, : 2])
        conf_loss_pos = mse_loss_function(obj_pre[ : , 4], obj_gt[ : , 4])
        conf_loss_neg = self.lambda_noobj * mse_loss_function(noobj_pre[ : , 4], torch.zeros(noobj_pre.size(0)).cuda())
        class_loss = cls_loss_function(obj_pre[ : , 5: ], obj_gt[ : , 5].long())
        total_loss = (xy_loss + wh_loss + conf_loss_neg + conf_loss_pos + class_loss)/batch_size
        return total_loss, {
            'xy_loss' : xy_loss / batch_size,
            'wh_loss' : wh_loss / batch_size,
            'conf_posi_loss': conf_loss_pos / batch_size,
            'conf_neg_loss': conf_loss_neg / batch_size,
            'class_loss' : class_loss / batch_size
        }
    
if __name__ == '__main__':
    # from dataset.voc import YOLOv2Dataset
    # image_root = '/home/asher/codes/python/yolo_series/DATASET/VOC/2007/JPEGImages'
    # label_txt = '/home/asher/codes/python/yolo_series/DATASET/train.txt'

    # img_size = 416
    # voc2007 = YOLOv2Dataset(image_root, label_txt, img_size)
    # from utils.tools import gt_creator
    # criterion = Loss()
    # for i in range(70, 75):
    #     _, target, path = voc2007[i]
    #     gt_tensor = gt_creator(int(img_size/ 32), target).cuda()
    #     gt_tensor = gt_tensor.unsqueeze(0)
    #     pre = torch.empty(1, 13, 13, 5, 25).uniform_(-1000, -900).cuda()
        
    #     pre[..., 5 : ] = 0
    #     loss, lossdict = criterion(pre, gt_tensor)
    #     print(lossdict)
    criterion = Loss()
    criterion.create_grid(13)
