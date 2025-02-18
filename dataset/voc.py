import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image
import os
from utils.tools import detection_collate

class YOLOv2Dataset(Dataset):
    def __init__(self, img_dir, label_txt):
        self.img_dir = img_dir
        self.label_path  = label_txt
        self.img = []
        self.xyxyclass = []  
        self.__parse_txt__()

    def __len__(self):
        return len(self.img)
    
    def __parse_txt__(self):
        with open(self.label_path) as f:
            lines = f.readlines()
            for line in lines:
                splitted = line.strip().split()
                fname = splitted[0]
                path = os.path.join(self.img_dir, fname)
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
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.img_dir, self.img[idx])
        image = Image.open(image_path).convert('RGB')
        xyxyc_norm = self.xyxyclass[idx]
        return image, xyxyc_norm

    
if __name__ == '__main__':
    image_root = '/home/asher/codes/python/self_yolo_imple_serires/DATASET/VOC/2007/JPEGImages'
    label_txt = '/home/asher/codes/python/self_yolo_imple_serires/DATASET/train.txt'

    voc2007 = YOLOv2Dataset(img_dir=image_root, label_txt=label_txt)

    dataloader = torch.utils.data.DataLoader(
                    dataset=voc2007, 
                    shuffle=False,
                    batch_size=64, 
                    num_workers=4,
                    collate_fn=detection_collate,
                    pin_memory=True,
                    drop_last=False
                    )
    for i , (img, gt) in enumerate(dataloader):
        print('iter{}, image shape {}, gt shape {}'.format(i, img.shape, gt.shape))


