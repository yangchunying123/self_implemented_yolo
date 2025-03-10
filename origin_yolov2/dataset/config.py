# config.py

# YOLOv2 with darknet-19
yolov2_d19_cfg = {
    # network
    'backbone': 'd19',
    # for multi-scale trick
    'train_size': 640,
    'val_size': 416,
    'random_size_range': [10, 19],
    # anchor size
    'anchor_size_voc': [[1.19, 1.98], [2.79, 4.59], [4.53, 8.92], [8.06, 5.29], [10.32, 10.65]],
    'anchor_size_coco': [[0.53, 0.79], [1.71, 2.36], [2.89, 6.44], [6.33, 3.79], [9.03, 9.74]],
    # train
    'lr_epoch': (150, 200),
    'max_epoch': 20,
    'ignore_thresh': 0.5
}
