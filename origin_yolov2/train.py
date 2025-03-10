from __future__ import division

import os
import argparse
import time
import torch
import torch.optim as optim
import utils.tools as tools
import random

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Detection')
    # basic
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('-bs', '--batch_size', default=24, type=int, 
                        help='Batch size for training')
    parser.add_argument('--lr', default=1e-3, type=float, 
                        help='initial learning rate')
    parser.add_argument('--wp_epoch', type=int, default=2,
                        help='The upper bound of warm-up')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='start epoch to train')
    parser.add_argument('-r', '--resume', default=None, type=str, 
                        help='keep training')
    parser.add_argument('--momentum', default=0.9, type=float, 
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, 
                        help='Weight decay for SGD')
    parser.add_argument('--num_workers', default=8, type=int, 
                        help='Number of workers used in dataloading')
    parser.add_argument('--num_gpu', default=1, type=int, 
                        help='Number of GPUs to train')
    parser.add_argument('--eval_epoch', type=int,
                            default=10, help='interval between evaluations')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('-v', '--version', default='yolo_v2',
                        help='yolov2_d19, yolov2_r50, yolov2_slim, yolov3, yolov3_spp, yolov3_tiny')
    parser.add_argument('-root', '--data_root', default='/mnt/share/ssd2/dataset',
                        help='dataset root')
    parser.add_argument('-d', '--dataset', default='voc',
                        help='voc or coco')
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False,
                        help='use multi-scale trick')      

    return parser.parse_args()

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train():
    args = parse_args()
    device = torch.device("cuda")

    from models.yolov2_d19 import YOLOv2D19 as yolo_net
    from dataset.config import yolov2_d19_cfg 
    cfg = yolov2_d19_cfg

    train_size = cfg['train_size']
    batch_size = args.batch_size

    image_root = '/home/asher/codes/python/self_yolo_imple_serires/DATASET/VOC/2007/JPEGImages'
    label_txt = '/home/asher/codes/python/self_yolo_imple_serires/DATASET/train.txt'
    from dataset.voc import YOLOv2Dataset

    dataset = YOLOv2Dataset(image_root, label_txt, train_size)
    dataloader = torch.utils.data.DataLoader(
                        dataset=dataset, 
                        shuffle=True,
                        collate_fn=tools.detection_collate,
                        batch_size=batch_size, 
                        num_workers=args.num_workers,
                        pin_memory=True,
                        drop_last=True
                        )

    anchor_size = cfg['anchor_size_voc'] if args.dataset == 'voc' else cfg['anchor_size_coco']
    net = yolo_net(device=device, 
                   input_size=train_size, 
                   num_classes=20, 
                   trainable=True, 
                   anchor_size=anchor_size)
    model = net.to(device).train()


    if args.tfboard:
        print('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter
        c_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        log_path = os.path.join('log/', args.dataset, c_time)
        os.makedirs(log_path, exist_ok=True)

        tblogger = SummaryWriter(log_path)
    
    # optimizer setup
    base_lr = (args.lr / 16) * batch_size
    tmp_lr = base_lr
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    max_epoch = cfg['max_epoch']
    epoch_size = len(dataloader)
    warmup = True


    best_loss = torch.tensor([1e10]).cuda()
    for epoch in range(args.start_epoch, max_epoch):
        total_loss = torch.tensor([0.0], dtype=float)
        print('Start Epoch: {}/{}'.format(epoch, max_epoch))
        
        if epoch in cfg['lr_epoch']:
            tmp_lr = tmp_lr * 0.1
            set_lr(optimizer, tmp_lr)
    
        for iter_i, (images, targets) in enumerate(dataloader):
            ni = iter_i + epoch * epoch_size
            # warmup
            if epoch < args.wp_epoch and warmup:
                nw = args.wp_epoch * epoch_size
                tmp_lr = base_lr * pow(ni / nw, 4)
                set_lr(optimizer, tmp_lr)

            elif epoch == args.wp_epoch and iter_i == 0 and warmup:
                # warmup is over
                warmup = False
                tmp_lr = base_lr
                set_lr(optimizer, tmp_lr)

        #     # multi-scale trick
            if iter_i % 10 == 0 and iter_i > 0 and args.multi_scale:
                # randomly choose a new size
                r = cfg['random_size_range']
                train_size = random.randint(r[0], r[1]) * 32
                model.set_grid(train_size)
            if args.multi_scale:
                # interpolate
                images = torch.nn.functional.interpolate(images, size=train_size, mode='bilinear', align_corners=False)
            
            targets = [label.tolist() for label in targets]
            targets = tools.gt_creator(input_size=train_size, 
                                           stride=net.stride, 
                                           label_lists=targets, 
                                           anchor_size=anchor_size
                                           )

            # to device
            images = images.float().to(device)
            targets = torch.tensor(targets).float().to(device)
            conf_loss, cls_loss, box_loss, iou_loss = model(images, target=targets)
            total_loss = conf_loss + cls_loss + box_loss + iou_loss
            if torch.isnan(total_loss):
                print('loss is nan !!')
                continue


            # backprop
            total_loss.backward()        
            optimizer.step()
            optimizer.zero_grad()

        if total_loss < best_loss:
            best_loss = total_loss        
            weight_name = '{}_epoch_{}.pth'.format(args.version, epoch + 1)
            torch.save(model.state_dict(), weight_name)   

    if args.tfboard:
        tblogger.close()






if __name__ == '__main__':
    train()
