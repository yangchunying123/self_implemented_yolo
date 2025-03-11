import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5):
        super(Loss, self).__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, pred_tensor, target_tensor):
        """ Compute loss for YOLO training.
        Args:
            pred_tensor: (Tensor) predictions, sized [n_batch, S, S, anchor, 5 + class]).
            target_tensor: (Tensor) targets, sized [n_batch, S, S, anchor, 5 + class], 5=len([offsetx, offsety, offsetw, offseth, conf].
        Returns:
            (Tensor): loss, sized [1, ].
        """
        assert pred_tensor.shape == target_tensor.shape, 'tensors should be in the same shape, shape of prediction {}, shape of gt {}'.format(pred_tensor.shape, target_tensor.shape)
        batch_size, N =  target_tensor.size(0), target_tensor.size(-1)
        obj_mask = (target_tensor[..., 4] > 0).unsqueeze(-1).expand_as(target_tensor)
        no_obj_mask = (target_tensor[..., 4] == 0).unsqueeze(-1).expand_as(target_tensor)
        obj_gt = target_tensor[obj_mask].view(-1, N)
        obj_pre = pred_tensor[obj_mask].view(-1, N)
        noobj_pre = pred_tensor[no_obj_mask].view(-1, N)

        cls_loss_function = nn.CrossEntropyLoss(reduction='sum')
        txty_loss_function = nn.BCEWithLogitsLoss(reduction='sum')
        twth_loss_function = nn.SmoothL1Loss(reduction='sum')
        conf_loss_function = nn.BCEWithLogitsLoss(reduction='sum')

        xy_loss = self.lambda_coord * txty_loss_function(obj_pre[:, : 2], obj_gt[:, : 2])
        wh_loss = self.lambda_coord * twth_loss_function(obj_pre[:, 2:4], torch.log(obj_gt[:, 2:4] + 1e-9))
        conf_loss_pos = conf_loss_function(obj_pre[ : , 4], obj_gt[ : , 4])
        conf_loss_neg = self.lambda_noobj * conf_loss_function(noobj_pre[ : , 4], torch.zeros(noobj_pre.size(0)).cuda())
        class_loss = cls_loss_function(obj_pre[ : , 5: ], obj_gt[ : , 5: ])
        total_loss = (xy_loss + wh_loss + conf_loss_neg + conf_loss_pos + class_loss)/batch_size
        return total_loss, {
            'xy_loss' : xy_loss,
            'wh_loss' : wh_loss,
            'conf_posi_loss': conf_loss_pos,
            'conf_neg_loss': conf_loss_neg,
            'class_loss' : class_loss
        }