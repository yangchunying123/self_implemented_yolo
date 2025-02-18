# config.py

STRIDE = 32
CLASS_NUM = 20
EPOCH = 220
BATCH_SIZE = 32
BASE_LR = 1e-3
CONF_THRESHOLD = 0.5
SUPRESSION = 0.4
CLASS =['aeroplane','bicycle', 'bird','boat','bottle',
        'bus','car','cat','chair', 'cow', 
        'diningtable', 'dog', 'horse', 'motorbike', 'person', 
        'pottedplant','sheep', 'sofa', 'train', 'tvmonitor']

# YOLOv2 with darknet-19
yolov2_d19_cfg = {
    # network
    'backbone': 'd19',
    # for multi-scale trick
    'train_size': 640,
    'val_size': 416,
    'random_size_range': [10, 19],
    # anchor size
    'anchor_size_voc': [[1.08, 1.19], [2.79, 3.42], [4.53, 6.12], [7.89, 5.23], [9.71, 9.92]],
    # train
    'lr_epoch': (150, 200),
    'ignore_thresh': 0.5
}
