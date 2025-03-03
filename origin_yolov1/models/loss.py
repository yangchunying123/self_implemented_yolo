import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import print

class Loss(nn.Module):
    def __init__(self, class_num, lambda_coord=5.0, lambda_noobj=0.5):
        super(Loss, self).__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.C = class_num

    def compute_iou(self, bbox1, bbox2):
        """ Compute the IoU (Intersection over Union) of two set of bboxes, each bbox format: [x1, y1, x2, y2].
        Args:
            bbox1: (Tensor) bounding bboxes, sized [N, 4].
            bbox2: (Tensor) bounding bboxes, sized [M, 4].
        Returns:
            (Tensor) IoU, sized [N, M].
        """
        N = bbox1.size(0)
        M = bbox2.size(0)

        # Compute left-top coordinate of the intersections
        lt = torch.max(
            bbox1[:, :2].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, :2].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Conpute right-bottom coordinate of the intersections
        rb = torch.min(
            bbox1[:, 2:].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, 2:].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Compute area of the intersections from the coordinates
        wh = rb - lt   # width and height of the intersection, [N, M, 2]
        wh[wh < 0] = 0 # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1] # [N, M]

        # Compute area of the bboxes
        area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1]) # [N, ]
        area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1]) # [M, ]
        area1 = area1.unsqueeze(1).expand_as(inter) # [N, ] -> [N, 1] -> [N, M]
        area2 = area2.unsqueeze(0).expand_as(inter) # [M, ] -> [1, M] -> [N, M]

        # Compute IoU from the areas
        union = area1 + area2 - inter # [N, M]
        iou = inter / union           # [N, M]

        return iou

    def forward(self, pred_tensor, target_tensor):
        """ Compute loss for YOLO training.
        Args:
            pred_tensor: (Tensor) predictions, sized [n_batch, S, S, Bx5+C]).
            target_tensor: (Tensor) targets, sized [n_batch, S, S, Bx5+C], 5=len([x, y, w, h, conf].
        Returns:
            (Tensor): loss, sized [1, ].
        """
        assert pred_tensor.shape == target_tensor.shape, 'tensors should be in the same shape, shape of prediction {}, shape of gt {}'.format(pred_tensor.shape, target_tensor.shape)
        batch_size, S, _, N = pred_tensor.size(0), pred_tensor.size(1), pred_tensor.size(2), pred_tensor.size(3) 
        B = (N - self.C) // 5 
        obj_mask = (target_tensor[..., 4] == 1).unsqueeze(-1).expand_as(target_tensor)
        no_obj_mask = (target_tensor[..., 4] < 1).unsqueeze(-1).expand_as(target_tensor) # actually, even one demension is label as "contains obj", but those predicted boxes that
                                                                                         # do not have largest iou with gt should also be regarded as no_obj. this should be dealed 
                                                                                         # with later
        my_gt, my_preds = self.txty2xy(target_tensor, pred_tensor)
        my_gt = my_gt[obj_mask].view(-1, N)             #(obj_num, N)
        my_preds = my_preds[obj_mask].view(-1, N)       #(obj_num, N)
        obj_num = my_gt.size(0)  # num of objects contains in a image/batch
        
        gt_objs =  target_tensor[obj_mask].view(-1, N).contiguous()     #(obj_num, 5)
        pred_objs = pred_tensor[obj_mask].view(-1, N).contiguous()
        preds_posi = torch.zeros(obj_num, 5).cuda()    #(obj_num, 5)

        extra_conf = []
        for  i in range(obj_num):
            gt = my_gt[i, : 4].unsqueeze(0)
            pred = my_preds[i, : 5 * B].contiguous().view(-1, 5) # pred [B, 5] (x, y, w, h, conf)
            gt_box = torch.zeros_like(gt).cuda()
            pred_box = torch.zeros(B, 4).cuda()
            gt_box[:,  :2] = gt[:, : 2] - 0.5 * gt[:, 2:4]
            gt_box[:, 2:4] = gt[:, : 2] + 0.5 * gt[:, 2:4]
            pred_box[:,  :2] = pred[:, : 2] - 0.5 * pred[:, 2:4]
            pred_box[:, 2:4] = pred[:, : 2] + 0.5 * pred[:, 2:4]
            iou = self.compute_iou(gt_box, pred_box)
            iou_value, index = iou.max(1)    
            iou_value, index = iou_value.item(), index.item()  
            gt_objs[i, 4] = iou_value
            for b in range(B):
                if b != index:
                    extra_conf.append(pred_objs[i, 5 * b + 4].item())
                else:
                    preds_posi[i] = pred_objs[i, 5 * b : 5 * b + 5]
        
        pre_nega_conf = pred_tensor[no_obj_mask].contiguous().view(-1, N)[ :, : 5 * B].contiguous().view(-1, 5)[..., 4] # single dimension tensor , shape should be ( S * S - obj_num) * B , curently
        pre_nega_conf = torch.cat([pre_nega_conf, torch.tensor(extra_conf.cuda())], dim=0)
        
        loss_xy = self.lambda_coord * F.mse_loss(preds_posi[..., : 2], gt_objs[..., : 2], reduction='sum')
        loss_wh = self.lambda_coord * F.mse_loss(torch.sqrt(torch.clamp(preds_posi[..., 2: 4], 0, 1)), torch.sqrt(gt_objs[..., 2: 4]), reduction='sum')
        loss_class = F.mse_loss(pred_objs[..., 5 * B :], gt_objs[..., 5 * B : ], reduction='sum')
        loss_conf_posi = F.mse_loss(preds_posi[..., 4], gt_objs[..., 4], reduction='sum')
        loss_conf_nega = self.lambda_noobj * F.mse_loss(pre_nega_conf, torch.zeros_like(pre_nega_conf), reduction='sum')
        # # Total loss
        loss = loss_xy + loss_wh + loss_conf_posi +  loss_conf_nega + loss_class
        loss = loss / float(batch_size)
        loss_dict = {
            'loss_xy': loss_xy, 
            'loss_wh' : loss_wh,
            'loss_class': loss_class, 
            'loss_conf_posi': loss_conf_posi,
            'loss_conf_nega': loss_conf_nega
        }
        return loss/float(batch_size)
    

    def txty2xy(self, gts, predictions):
        my_gts, my_preds = gts.clone(), predictions.clone()
        batch_size, S, _, N = predictions.size(0), predictions.size(1), predictions.size(2), predictions.size(3) 
        B = (N - self.C) // 5 
        from origin_yolov1.utils.tools import create_grid
        grid = create_grid(S, batch=batch_size)
        for b in range(B):
            my_gts[..., 5 *b : 5 * b + 2] = (my_gts[..., 5 *b : 5 * b + 2] + grid) / float(S)
            my_preds[..., 5 *b : 5 * b + 2] = (my_preds[..., 5 *b : 5 * b + 2] + grid)/float(S)
        return my_gts, my_preds

if __name__ == '__main__':
    t = torch.rand(1, 7, 7, 30)
    t[..., 5: 10] = 0
    import random
    for _ in range(5):
        i = random.randint(0, 6)
        j = random.randint(0, 6)
        t[:, j, i, 4] = 1
    pre = torch.rand(1, 7, 7, 30)
    myloss = Loss(20)
    myloss(pre, t)