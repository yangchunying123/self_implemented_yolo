from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import os
from PIL import Image
from torch.utils.data import DataLoader
class ImageTransfrom:
    def __init__(self, image_size):
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def __call__(self, raw_img):
        return self.transform(raw_img)
class YOLOv2Dataset(Dataset):
    def __init__(self, img_dir, label_txt, img_size):
        self.img_dir = img_dir
        self.img = []
        self.xywhclass = []  # ycy implemented normalized
        self.image_trans = ImageTransfrom(img_size)

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
                    cxywh.append([(x1 + x2 ) /2 / w, (y1 + y2)/2/h, (x2 - x1)/w, (y2 - y1)/h, c])
                self.xywhclass.append(cxywh)

    def __len__(self):
        return len(self.img)

    def pull_item(self, idx):
        img_path = self.img[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.image_trans(img)
        return img, self.xywhclass[idx], img_path

    def __getitem__(self, index):
        return self.pull_item(index)
    
if __name__ == '__main__':
    image_root = '/home/asher/codes/python/yolo_series/DATASET/VOC/2007/JPEGImages'
    label_txt = '/home/asher/codes/python/yolo_series/DATASET/train.txt'

    img_size = 416
    voc2007 = YOLOv2Dataset(image_root, label_txt, img_size)
    # from utils.tools import visualize_gt_tensor, gt_creator
    # for i in range(70, 100):
    #     _, target, path = voc2007[i]
    #     gt_tensor = gt_creator(int(img_size/ 32), target)
    #     visualize_gt_tensor(path, gt_tensor)

    from utils.tools import yolo_collefn_in, batch_gt_tensor_creator
    dataloader = DataLoader(
                    dataset=voc2007, 
                    shuffle=False,
                    batch_size=16, 
                    num_workers=4,
                    collate_fn=yolo_collefn_in,
                    pin_memory=True,
                    drop_last=False
                    )
    for i , (img, labels) in enumerate(dataloader):
        gts = batch_gt_tensor_creator(int(img_size/32), labels)
        print(img.shape, gts.shape)

