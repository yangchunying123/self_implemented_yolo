# config.py

# YOLOv2 with darknet-19
yolov2_d19_cfg = {
    # network
    'backbone': 'd19',
    # for multi-scale trick
    'train_size': 416,
    'val_size': 416,
    'random_size_range': [13, 13],
    # anchor size
    'anchor_size_voc': [[1.08, 1.19], [3.42, 4.41], [6.63, 11.38], [9.42, 5.11], [16.62, 10.52]],
    'anchor_size_coco': [[0.53, 0.79], [1.71, 2.36], [2.89, 6.44], [6.33, 3.79], [9.03, 9.74]],
    'anchor_size_voc_old': [[1.19, 1.98], [2.79, 4.59], [4.53, 8.92], [8.06, 5.29], [10.32, 10.65]],
    # train
    'ignore_thresh': 0.5,
    'class_num': 20,
    'stride' : 32
}

yolov2_train_cfg = {
    'batch_size' : 64,
    'lr': 1e-4,
    'max_epoch': 80,
    'loss_save_iter' : 5,
    'loss_save_iter_val': 100
}
