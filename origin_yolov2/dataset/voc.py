import torch
from torch.utils.data import Dataset
import cv2
import os
from dataset.augmentations import SSDAugmentation
import numpy as np

class YOLOv2Dataset(Dataset):
    def __init__(self, img_dir, label_txt, img_size):
        self.image_bbox_label_trans = SSDAugmentation(size=img_size)
        self.img_dir = img_dir
        self.img = []
        self.xyxyclass = []  # ycy implemented normalized

        with open(label_txt) as f:
            lines = f.readlines()
            for line in lines:
                splitted = line.strip().split()
                fname = splitted[0]
                path = os.path.join(img_dir, fname)
                if not os.path.exists(path):
                    continue

                self.img.append(path)
                img = cv2.imread(path)
                h, w, _ = img.shape
                cxywh = []
                for i in range((len(splitted) - 1) // 5):
                    x1 = int(splitted[5*i + 1])
                    y1 = int(splitted[5*i + 2])
                    x2 = int(splitted[5*i + 3])
                    y2 = int(splitted[5*i + 4])
                    c  =   int(splitted[5*i + 5])
                    cxywh.append([x1/ w, y1/h, x2/w, y2/h, c])
                self.xyxyclass.append(cxywh)


        


    def __len__(self):
        return len(self.img)

    def pull_item(self, idx):
        img_path = self.img[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        target = np.array(self.xyxyclass[idx])
        img, bbox, label = self.image_bbox_label_trans(img, target[:, :4], target[:, 4])
        img, bbox, label = torch.from_numpy(img).permute(2, 0, 1).float(), torch.from_numpy(bbox), torch.from_numpy(label).unsqueeze(1)
        target = torch.cat([bbox, label], dim=1).float()
        return img, target

    def __getitem__(self, index):
        return self.pull_item(index)
    
if __name__ == '__main__':
    from utils.tools import detection_collate
    image_root = '/home/asher/codes/python/self_yolo_imple_serires/DATASET/VOC/2007/JPEGImages'
    label_txt = '/home/asher/codes/python/self_yolo_imple_serires/DATASET/train.txt'

    voc2007 = YOLOv2Dataset(image_root, label_txt, 416)

    # dataloader = torch.utils.data.DataLoader(
    #                 dataset=voc2007, 
    #                 shuffle=False,
    #                 batch_size=16, 
    #                 num_workers=4,
    #                 collate_fn=detection_collate,
    #                 pin_memory=True,
    #                 drop_last=True
    #                 )
    # for i , (img, gt) in enumerate(dataloader):
    #     print(i, img.shape)
    #     print(type(gt))
    #     print('image', gt[0].shape, img[1].shape)
    #     break


