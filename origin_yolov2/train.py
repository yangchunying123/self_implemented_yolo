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
    from dataset.voc import YOLOv2Dataset
    from torch.utils.data import DataLoader
    from utils.tools import yolo_collefn_in
    image_dir = '/home/asher/codes/python/yolo_series/DATASET/VOC/2007/JPEGImages'
    train_label = '/home/asher/codes/python/yolo_series/DATASET/train.txt'
    val_label = '/home/asher/codes/python/yolo_series/DATASET/val.txt'

    train_dataset = YOLOv2Dataset(image_dir, train_label, image_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8,  collate_fn=yolo_collefn_in)
    val_dataset = YOLOv2Dataset(image_dir, val_label, image_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8,  collate_fn=yolo_collefn_in)

    print('Number of training images: ', len(train_dataset))
    print('Number of validation images: ', len(val_dataset))
    return {'train': train_loader, 'val': val_loader}

def train():
    from dataset.config import yolov2_train_cfg
    from models.yolov2_d19 import YOLOv2D19 as yolo_net
    from dataset.config import yolov2_d19_cfg as cfg
    from utils.tools import batch_gt_tensor_creator, batch_image_tensor_creator
    from models.loss import Loss

    device = torch.device("cuda")

    image_size = cfg['train_size']
    batch_size = yolov2_train_cfg['batch_size']
    loader_dict = get_data_loader(batch_size, image_size)

    anchor_size = cfg['anchor_size_voc'] 
    net = yolo_net(device=device, input_size=image_size, num_classes=20, anchor_size=anchor_size)
    model = net.to(device).train()
    # optimizer = torch.optim.Adam(model.parameters(), lr=yolov2_train_cfg['lr'])
    optimizer = torch.optim.SGD([
    {'params': model.backbone.parameters(), 'lr': 3e-5},
    {'params': model.convsets_1.parameters(), 'lr': 1e-3,
     'params': model.route_layer.parameters(), 'lr': 1e-3,
     'params': model.reorg.parameters(), 'lr': 1e-3,
     'params': model.convsets_2.parameters(), 'lr': 1e-3,
     'params': model.pred.parameters(), 'lr': 1e-3}
], momentum=0.9, weight_decay=5e-4)

    criterion = Loss().to(device)

    
    c_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    log_path = os.path.join('results', c_time)
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_path)
    
    
    iters_per_epoch_train = len(loader_dict['train'])
    best_loss = torch.tensor([1e10]).cuda()

    grid_num = int(cfg['train_size']/ cfg['stride'])
    for epoch in range(yolov2_train_cfg['max_epoch']):
        total_train_loss = torch.tensor([0.0], dtype=float).to(device)
        total_batch_train = 0  
        for iter_i, (images, targets) in enumerate(loader_dict['train']):
            if iter_i % 10 == 0 and iter_i > 0:
                r = cfg['random_size_range']
                grid_num = random.randint(r[0], r[1])
            images = batch_image_tensor_creator(grid_num, images).to(device)
            targets = batch_gt_tensor_creator(grid_num, targets).to(device)
            preds = model(images)
            loss, loss_dict = criterion(preds, targets)
            if torch.isnan(loss):
                print('Loss of epoch {}/{}, iter {} is nan'.format(epoch, yolov2_train_cfg['max_epoch'], iter_i))
                continue
            
            ith_iter = epoch * iters_per_epoch_train + iter_i
            if ith_iter % yolov2_train_cfg['loss_save_iter'] == 0:
                writer.add_scalar('train loss',loss, ith_iter)
                for k, v in loss_dict.items():
                    writer.add_scalar(k, v.item(), ith_iter)

            total_train_loss += loss
            total_batch_train += images.size(0)
            loss.backward()        
            optimizer.step()
            optimizer.zero_grad()
        
        
        grid_num = int(cfg['train_size']/ cfg['stride'])
        val_loss = torch.tensor([0.0], dtype=float).to(device)
        total_batch = 0
        model.eval()
        for i, (imgs, targets) in enumerate(loader_dict['val']):
            batch_size_this_iter = imgs.size(0)
            imgs = imgs.cuda()
            with torch.no_grad():
                preds = model(imgs)
            targets = batch_gt_tensor_creator(grid_num, targets).to(device)
            loss, _ = criterion(preds, targets)
            loss_this_iter = loss.item()
            val_loss += loss_this_iter * batch_size_this_iter
            total_batch += batch_size_this_iter
        val_loss /= float(total_batch)

        if best_loss > val_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(log_path, './model_best.pth'))

        writer.add_scalar('val loss', val_loss, epoch + 1)
        print('Epoch [%d/%d], Train Loss: %.4f, Val Loss: %.4f, Best Val Loss: %.4f' % (epoch + 1, yolov2_train_cfg['max_epoch'], total_train_loss/total_batch_train, val_loss, best_loss))
    writer.close()

if __name__ == '__main__':
    train()
