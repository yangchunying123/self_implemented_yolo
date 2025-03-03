import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import cv2
from rich import print
from PIL import Image

class ImageTransfrom:
    def __init__(self, image_size):
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def __call__(self, raw_img):
        return self.transform(raw_img)

class VOCDataset(Dataset):
    def __init__(self, image_dir, label_txt, image_trans, target_trans = None, grid_size=7, num_bboxes=2, num_classes=20):
        self.S = grid_size
        self.B = num_bboxes
        self.C = num_classes
        self.paths, self.boxes, self.labels = [], [], []   
        self.image_trans = image_trans
        self.target_trans = target_trans
        self.stride = 32

        with open(label_txt) as f:
            lines = f.readlines()

        for line in lines:
            splitted = line.strip().split()
            fname = splitted[0]
            path = os.path.join(image_dir, fname)
            self.paths.append(path)
            h, w , _ = cv2.imread(path).shape

            num_boxes = (len(splitted) - 1) // 5
            box, label = [], []
            for i in range(num_boxes):
                x1 = float(splitted[5*i + 1])
                y1 = float(splitted[5*i + 2])
                x2 = float(splitted[5*i + 3])
                y2 = float(splitted[5*i + 4])
                c  =   int(splitted[5*i + 5])
                x = (x1 + x2)/2/w
                y = (y1 + y2)/2/h
                width = (x2 - x1)/w
                height = (y2 - y1)/h
                box.append([x, y, width, height])
                label.append(c)
            self.boxes.append(box)
            self.labels.append(label)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('RGB')
        target = self.encode(self.boxes[idx], self.labels[idx]) # [S, S, 5 x B + C]
        img = self.image_trans(img)
        if self.target_trans is not None:
            target = self.target_trans(target)

        return img, target # 5 x B + C  [tx, ty, tw, th](between 0 -1(normalized))

    def __len__(self):
        return len(self.paths)

    def encode(self, boxes, labels):
        """ Encode box coordinates and class labels as one target tensor.
        Args:
            boxes: (tensor) [[x1, y1, x2, y2]_obj1, ...], normalized from 0.0 to 1.0 w.r.t. image width/height.
            labels: (tensor) [c_obj1, c_obj2, ...]
        Returns:
            An encoded tensor sized [S, S, 5 x B + C], 5=(gridx, gridy, nolmal_w, normal_h, conf)
        """
        S, B, C = self.S, self.B, self.C
        N = 5 * B + C
        target = torch.zeros(S, S, N)
        for box, label in zip(boxes, labels):
            xc, yc, w, h = float(box[0] * S), float(box[1] * S), box[2], box[3]
            i, j = int(xc), int(yc) 
            target[j, i,  : 4] = torch.tensor([xc - i, yc - j, w, h])
            target[j, i, 4    ] = 1.0
            target[j, i, 5*B + label] = 1.0
        return target


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    image_dir = '/home/asher/codes/python/self_yolo_imple_serires/DATASET/VOC/2007/JPEGImages'
    train_label = '/home/asher/codes/python/self_yolo_imple_serires/DATASET/train.txt'
    image_size = 448

    train_dataset = VOCDataset(image_dir, train_label, image_trans=ImageTransfrom(image_size))
    for i in range(440, 452):
        from utils.tools import visualize_encoded_tensor
        img, target = train_dataset[i]
        cvimg = cv2.imread(train_dataset.paths[i])
        visualize_encoded_tensor(cvimg, target)