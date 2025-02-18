import torch
import torch.optim as optim
from dataset.voc import YOLOv2Dataset
from utils.tools import detection_collate
from dataset.config import EPOCH, BATCH_SIZE, BASE_LR, yolov2_d19_cfg

image_root = '/home/asher/codes/python/self_yolo_imple_serires/DATASET/VOC/2007/JPEGImages'
train_txt = '/home/asher/codes/python/self_yolo_imple_serires/DATASET/train.txt'
eval_txt = '/home/asher/codes/python/self_yolo_imple_serires/DATASET/val.txt'

def prepare_data():
    voc2007_train = YOLOv2Dataset(img_dir=image_root, label_txt=train_txt)
    voc2007_eval = YOLOv2Dataset(img_dir=image_root, label_txt=eval_txt)
    dataloader_train = torch.utils.data.DataLoader(dataset=voc2007_train,  shuffle=True, batch_size=BATCH_SIZE, 
                                                   num_workers=4, collate_fn=detection_collate, pin_memory=True, drop_last=False)
    dataloader_eval = torch.utils.data.DataLoader(dataset=voc2007_eval,  shuffle=False, batch_size=BATCH_SIZE, 
                                                   num_workers=4, collate_fn=detection_collate, pin_memory=True, drop_last=False)
    return {'train': dataloader_train,  'eval': dataloader_eval}

def train():
    import time
    from models.yolo import YOLOLOSS, YOLO
    device = torch.device('cuda')
    loader_dict = prepare_data()

    model = YOLO(pretrained_backbone=True).to(device)
    criterion = YOLOLOSS(lambda_noobj=0.1)

    optimizer = optim.SGD(model.parameters(), lr=BASE_LR, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epoch for epoch in yolov2_d19_cfg['lr_epoch']], gamma=0.1)


    best_val_loss = float('inf')
    for epoch in range(EPOCH):
        start = time.time()
        train_loss = 0.0
        val_loss = 0.0

        model.train()
        for iter, (imgs, gts) in enumerate(loader_dict['train']):
            imgs, gts = imgs.to(device), gts.to(device)
            preds = model(imgs)
            loss = criterion(preds, gts)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item() * imgs.size(0)
        scheduler.step()

        model.eval()
        for i, (imgs, targets) in enumerate(loader_dict['eval']):
            imgs, targets = imgs.to(device), targets.to(device)
            with torch.no_grad():
                preds = model(imgs)
            loss = criterion(preds, targets)
            loss_this_iter = loss.item()
            val_loss += loss_this_iter * imgs.size(0)

        train_loss /= len(loader_dict['train'])
        val_loss /= len(loader_dict['eval'])


        if best_val_loss > val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), './model_best.pth')
        
        end = time.time()
        print('Epoch {}/{}, use {} seconds, average train loss is {}, average eval loss is {}'.format(epoch + 1, EPOCH, end - start, train_loss, val_loss))

if __name__ == '__main__':
    train()
        
