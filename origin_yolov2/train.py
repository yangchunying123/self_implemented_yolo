from __future__ import division

import os
import time
import torch
import torch.optim as optim
import utils.tools as tools
from torch.utils.tensorboard import SummaryWriter
import random

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_data_loader(batch_size, image_size):
    from dataset.voc import YOLOv2Dataset, DataAugment
    from torch.utils.data import DataLoader
    from utils.tools import yolo_collefn_in
    image_dir = '/home/asher/codes/python/yolo_series/DATASET/VOC/2007/JPEGImages'
    train_label = '/home/asher/codes/python/yolo_series/DATASET/train.txt'
    val_label = '/home/asher/codes/python/yolo_series/DATASET/val.txt'

    train_dataset = YOLOv2Dataset(image_dir, train_label, image_size, dataaugment=DataAugment(image_size))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8,  collate_fn=yolo_collefn_in)
    val_dataset = YOLOv2Dataset(image_dir, val_label, image_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8,  collate_fn=yolo_collefn_in)

    print('Number of training images: ', len(train_dataset))
    print('Number of validation images: ', len(val_dataset))
    return {'train': train_loader, 'val': val_loader}, (len(train_dataset), len(val_dataset))

def train():
    from dataset.config import yolov2_train_cfg
    from models.yolov2_d19 import YOLOv2D19 as yolo_net
    from dataset.config import yolov2_d19_cfg as cfg
    from utils.tools import batch_gt_tensor_creator, batch_image_tensor_creator
    from models.loss import Loss

    device = torch.device("cuda")

    image_size = cfg['train_size']
    batch_size = yolov2_train_cfg['batch_size']
    loader_dict, (train_sample_num, val_sample_num) = get_data_loader(batch_size, image_size)

    net = yolo_net(device=device)
    model = net.to(device).train()
    criterion = Loss().to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=yolov2_train_cfg['lr'])
#     optimizer = torch.optim.SGD([
#      {'params': model.backbone.parameters(), 'lr': 6e-5},
#      {'params': model.convsets_1.parameters(), 'lr': 1e-3},
#     #  {'params': model.route_layer.parameters(), 'lr': 1e-3},
#     #  {'params': model.reorg.parameters(), 'lr': 1e-3},
#     #  {'params': model.convsets_2.parameters(), 'lr': 1e-3},
#      {'params': model.pred.parameters(), 'lr': 1e-3}
# ], momentum=0.9, weight_decay=5e-3)
    optimizer = torch.optim.SGD(model.parameters(), lr=yolov2_train_cfg['lr'], momentum=0.9, weight_decay=5e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)

    
    c_time = time.strftime('%Y_%m_%d_%H:%M:%S',time.localtime(time.time()))
    log_path = os.path.join('results', str(c_time))
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_path)
    with open(os.path.join(log_path, 'record.txt'), "w") as f:
        pass  
    
    iters_per_epoch_train = len(loader_dict['train'])
    iters_per_epoch_val = len(loader_dict['val'])
    best_loss = torch.tensor([1e10]).cuda()

    grid_num = int(cfg['train_size']/ cfg['stride'])
    for epoch in range(yolov2_train_cfg['max_epoch']):
        total_train_loss = torch.tensor([0.0], dtype=float).to(device)  
        val_loss = torch.tensor([0.0], dtype=float).to(device)
        
        for iter_i, (images, targets) in enumerate(loader_dict['train']):
            if iter_i % 10 == 0 and iter_i > 0:
                r = cfg['random_size_range']
                grid_num = random.randint(r[0], r[1])
            images = batch_image_tensor_creator(grid_num, images).to(device)
            targets = batch_gt_tensor_creator(grid_num, targets).to(device)
            preds = model(images)
            loss, loss_dict = criterion(preds, targets)
            if torch.isnan(loss):
                print('Loss of epoch {}/{}, iter {} is nan'.format(epoch, yolov2_train_cfg['max_epoch'], iter_i), loss_dict)
                continue
            
            ith_iter = epoch * iters_per_epoch_train + iter_i
            if ith_iter % yolov2_train_cfg['loss_save_iter'] == 0:
                writer.add_scalar('train loss',loss, ith_iter)
                for k, v in loss_dict.items():
                    writer.add_scalar(k , v.item(), ith_iter)

            total_train_loss += loss * images.size(0)
            optimizer.zero_grad()
            loss.backward()        
            optimizer.step()
            scheduler.step()
                    
        
        grid_num = int(cfg['train_size']/ cfg['stride'])
        model.eval()
        for i, (imgs, targets) in enumerate(loader_dict['val']):
            batch_size_this_iter = imgs.size(0)
            imgs = imgs.cuda()
            with torch.no_grad():
                preds = model(imgs)
            targets = batch_gt_tensor_creator(grid_num, targets).to(device)
            loss, val_loss_dict = criterion(preds, targets)
            val_loss += loss.item() * batch_size_this_iter
            ith_iter = epoch * iters_per_epoch_val + i
            if ith_iter % yolov2_train_cfg['loss_save_iter_val'] == 0:
                for k, v in val_loss_dict.items():
                    writer.add_scalar(k + '_val', v.item(), ith_iter)

        val_loss /= float(val_sample_num)

        if best_loss > val_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(log_path, './model_best.pth'))

        writer.add_scalar('val loss', val_loss, epoch + 1)
        print('Epoch [%d/%d], Train Loss: %.4f, Val Loss: %.4f, Best Val Loss: %.4f' % (epoch + 1, yolov2_train_cfg['max_epoch'], total_train_loss/train_sample_num, val_loss, best_loss))
    writer.close()

if __name__ == '__main__':
    train()