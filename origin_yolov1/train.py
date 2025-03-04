import torch
from torch.autograd import Variable
import os
import numpy as np
import math
from tensorboardX import SummaryWriter
from rich import print

print_freq = 5
tb_log_freq = 5

init_lr = 1e-3
base_lr = 0.01
momentum = 0.9
weight_decay = 5.0e-4
num_epochs = 1
batch_size = 64
image_size = 416

def update_lr(optimizer, epoch, burnin_base, burnin_exp=4.0):
    if epoch == 0:
        lr = init_lr + 0.1* (base_lr - init_lr) * math.pow(burnin_base, burnin_exp)
    # elif epoch == 1:
    #     lr = base_lr
    elif epoch == 25:
        lr = 0.001
    elif epoch == 105:
        lr = 0.0001
    else:
        return

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_data_loader():
    from datas.voc import VOCDataset, ImageTransfrom
    from torch.utils.data import DataLoader
    image_dir = '/home/asher/codes/python/yolo_series/DATASET/VOC/2007/JPEGImages'
    train_label = '/home/asher/codes/python/yolo_series/DATASET/train.txt'
    val_label = '/home/asher/codes/python/yolo_series/DATASET/val.txt'

    train_dataset = VOCDataset(image_dir, train_label, image_trans=ImageTransfrom(image_size), grid_size=int(image_size/32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_dataset = VOCDataset(image_dir, val_label, image_trans=ImageTransfrom(image_size), grid_size=int(image_size/32))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    print('Number of training images: ', len(train_dataset))
    print('Number of validation images: ', len(val_dataset))
    return {'train': train_loader, 'val': val_loader}


def train():
    from datetime import datetime
    from models.yolo import YOLOV1
    from models.loss import Loss
    dataloader_dict = get_data_loader()
    model = YOLOV1()
    yolo = model.cuda()
    criterion = Loss(20).cuda()
    optimizer = torch.optim.SGD(yolo.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay)

    log_dir = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('results', log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    logfile = open(os.path.join(log_dir, 'log.txt'), 'w')

    best_val_loss = np.inf

    for epoch in range(num_epochs):
        yolo.train()
        total_loss = 0.0
        total_batch = 0

        for i, (imgs, targets) in enumerate(dataloader_dict['train']):
            update_lr(optimizer, epoch, float(i) / float(len(dataloader_dict['train']) - 1))
            lr = get_lr(optimizer)
            batch_size_this_iter = imgs.size(0)
            imgs = Variable(imgs)
            targets = Variable(targets)
            imgs, targets = imgs.cuda(), targets.cuda()

            # Forward to compute loss.
            preds = yolo(imgs)
            loss = criterion(preds, targets)
            loss_this_iter = loss.item()
            total_loss += loss_this_iter * batch_size_this_iter
            total_batch += batch_size_this_iter

            # Backward to update model weight.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % print_freq == 0:
                print('Epoch [%d/%d], Iter [%d/%d], LR: %.6f, Loss: %.4f, Average Loss: %.4f' % (epoch, num_epochs, i, len(dataloader_dict['train']), lr, loss_this_iter, total_loss / float(total_batch)))
            n_iter = epoch * len(dataloader_dict['train']) + i
            if n_iter % tb_log_freq == 0:
                writer.add_scalar('train/loss', loss_this_iter, n_iter)
                writer.add_scalar('lr', lr, n_iter)
        
        yolo.eval()
        val_loss = 0.0
        total_batch = 0

        for i, (imgs, targets) in enumerate(dataloader_dict['val']):
            batch_size_this_iter = imgs.size(0)
            imgs = Variable(imgs)
            targets = Variable(targets)
            imgs, targets = imgs.cuda(), targets.cuda()
            with torch.no_grad():
                preds = yolo(imgs)
            loss = criterion(preds, targets)
            loss_this_iter = loss.item()
            val_loss += loss_this_iter * batch_size_this_iter
            total_batch += batch_size_this_iter
        val_loss /= float(total_batch)

        logfile.writelines(str(epoch + 1) + '\t' + str(val_loss) + '\n')
        logfile.flush()
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            torch.save(yolo.state_dict(), os.path.join(log_dir, './model_best.pth'))

        print('Epoch [%d/%d], Val Loss: %.4f, Best Val Loss: %.4f' % (epoch + 1, num_epochs, val_loss, best_val_loss))
        writer.add_scalar('test/loss', val_loss, epoch + 1)

    writer.close()
    logfile.close()

if __name__ == '__main__':
    train()
    # test_forward_iteration()