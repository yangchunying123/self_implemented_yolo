import torch
import os
import cv2
from rich import print
from models.yolov2_d19 import YOLOv2D19

# VOC class names and BGR color.
VOC_CLASS_BGR = {
    'aeroplane': (128, 0, 0),
    'bicycle': (0, 128, 0),
    'bird': (128, 128, 0),
    'boat': (0, 0, 128),
    'bottle': (128, 0, 128),
    'bus': (0, 128, 128),
    'car': (128, 128, 128),
    'cat': (64, 0, 0),
    'chair': (192, 0, 0),
    'cow': (64, 128, 0),
    'diningtable': (192, 128, 0),
    'dog': (64, 0, 128),
    'horse': (192, 0, 128),
    'motorbike': (64, 128, 128),
    'person': (192, 128, 128),
    'pottedplant': (0, 64, 0),
    'sheep': (128, 64, 0),
    'sofa': (0, 192, 0),
    'train': (128, 192, 0),
    'tvmonitor': (0, 64, 128)
}


def visualize_boxes(image_bgr, boxes, class_names, probs, name_bgr_dict=None, line_thickness=2):
    if name_bgr_dict is None:
        name_bgr_dict = VOC_CLASS_BGR

    image_boxes = image_bgr.copy()
    for box, class_name, prob in zip(boxes, class_names, probs):
        # Draw box on the image.
        left_top, right_bottom = box
        left, top = int(left_top[0]), int(left_top[1])
        right, bottom = int(right_bottom[0]), int(right_bottom[1])
        bgr = name_bgr_dict[class_name]
        cv2.rectangle(image_boxes, (left, top), (right, bottom), bgr, thickness=line_thickness)

        # Draw text on the image.
        text = '%s %.2f' % (class_name, prob)
        size, baseline = cv2.getTextSize(text,  cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2)
        text_w, text_h = size

        x, y = left, top
        x1y1 = (x, y)
        x2y2 = (x + text_w + line_thickness, y + text_h + line_thickness + baseline)
        cv2.rectangle(image_boxes, x1y1, x2y2, bgr, -1)
        cv2.putText(image_boxes, text, (x + line_thickness, y + 2*baseline + line_thickness),
            cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 255, 255), thickness=1, lineType=8)

    return image_boxes


class YOLODetector:
    
    def __init__(self, model_path, class_name_list=None, conf_thresh=0.1, prob_thresh=0.1, nms_thresh=0.5, gpu_id=0):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print("Loading YOLO model...")
        self.yolo = YOLOv2D19(device = torch.device('cuda'))
        self.yolo.load_state_dict(torch.load(model_path))
        print("Done loading!")
        self.yolo.eval()
        self.class_name_list = class_name_list if (class_name_list is not None) else list(VOC_CLASS_BGR.keys())
        self.conf_thresh = conf_thresh
        self.prob_thresh = prob_thresh
        self.nms_thresh = nms_thresh

        dummy_input = torch.rand(16, 3, 416, 416).cuda()
        for i in range(10):
            self.yolo(dummy_input)

    def detect(self, image_path, image_size=416):
        from dataset.voc import ImageTransfrom
        from PIL import Image
        img = Image.open(image_path).convert('RGB')
        w, h = img.size
        imgtrans = ImageTransfrom(image_size)
        img = imgtrans(img).cuda()
        img = img.unsqueeze(0)

        with torch.no_grad():
            pred_tensor = self.yolo(img)
        pred_tensor = pred_tensor.cpu().data
        pred_tensor = pred_tensor.squeeze(0) # squeeze batch dimension.

        # Get detected boxes_detected, labels, confidences, class-scores.
        boxes_normalized_all, class_labels_all, confidences_all, class_scores_all = self.decode(pred_tensor)
        if boxes_normalized_all.size(0) == 0:
            return [], [], [] # if no box found, return empty lists.

        # Apply non maximum supression for boxes of each class.
        boxes_normalized, class_labels, probs = [], [], []

        for class_label in range(len(self.class_name_list)):
            mask = (class_labels_all == class_label)
            if torch.sum(mask) == 0:
                continue # if no box found, skip that class.

            boxes_normalized_masked = boxes_normalized_all[mask]
            class_labels_maked = class_labels_all[mask]
            confidences_masked = confidences_all[mask]
            class_scores_masked = class_scores_all[mask]

            ids = self.nms(boxes_normalized_masked, confidences_masked)

            boxes_normalized.append(boxes_normalized_masked[ids])
            class_labels.append(class_labels_maked[ids])
            probs.append(confidences_masked[ids] * class_scores_masked[ids])

        boxes_normalized = torch.cat(boxes_normalized, 0)
        class_labels = torch.cat(class_labels, 0)
        probs = torch.cat(probs, 0)

        # Postprocess for box, labels, probs.
        boxes_detected, class_names_detected, probs_detected = [], [], []
        for b in range(boxes_normalized.size(0)):
            box_normalized = boxes_normalized[b]
            class_label = class_labels[b]
            prob = probs[b]

            x1, x2 = w * box_normalized[0], w * box_normalized[2] # unnormalize x with image width.
            y1, y2 = h * box_normalized[1], h * box_normalized[3] # unnormalize y with image height.
            boxes_detected.append(((x1, y1), (x2, y2)))

            class_label = int(class_label) # convert from LongTensor to int.boxes_detected
            class_name = self.class_name_list[class_label]
            class_names_detected.append(class_name)

            prob = float(prob) # convert from Tensor to float.
            probs_detected.append(prob)

        return boxes_detected, class_names_detected, probs_detected

    def decode(self, pred_tensor):
        H, W, anchor_num, N = pred_tensor.size(0), pred_tensor.size(1), pred_tensor.size(2), pred_tensor.size(-1)
        boxes, labels, confidences, class_scores = [], [], [], []
        for j in range(H): 
            for i in range(W): 
                for anch in  range(anchor_num):
                    conf = pred_tensor[j, i, anch, 4]
                    targetiou, index = torch.max(pred_tensor[j, i, anch, 5: ], dim=1)

          

                    

    def nms(self, boxes, scores):
        """ Apply non maximum supression.
        Args:
        Returns:
        """
        threshold = self.nms_thresh

        x1 = boxes[:, 0] # [n,]
        y1 = boxes[:, 1] # [n,]
        x2 = boxes[:, 2] # [n,]
        y2 = boxes[:, 3] # [n,]
        areas = (x2 - x1) * (y2 - y1) # [n,]

        _, ids_sorted = scores.sort(0, descending=True) # [n,]
        ids = []
        while ids_sorted.numel() > 0:
            # Assume `ids_sorted` size is [m,] in the beginning of this iter.

            i = ids_sorted.item() if (ids_sorted.numel() == 1) else ids_sorted[0]
            ids.append(i)

            if ids_sorted.numel() == 1:
                break # If only one box is left (i.e., no box to supress), break.

            inter_x1 = x1[ids_sorted[1:]].clamp(min=x1[i]) # [m-1, ]
            inter_y1 = y1[ids_sorted[1:]].clamp(min=y1[i]) # [m-1, ]
            inter_x2 = x2[ids_sorted[1:]].clamp(max=x2[i]) # [m-1, ]
            inter_y2 = y2[ids_sorted[1:]].clamp(max=y2[i]) # [m-1, ]
            inter_w = (inter_x2 - inter_x1).clamp(min=0) # [m-1, ]
            inter_h = (inter_y2 - inter_y1).clamp(min=0) # [m-1, ]

            inters = inter_w * inter_h # intersections b/w/ box `i` and other boxes, sized [m-1, ].
            unions = areas[i] + areas[ids_sorted[1:]] - inters # unions b/w/ box `i` and other boxes, sized [m-1, ].
            ious = inters / unions # [m-1, ]

            # Remove boxes whose IoU is higher than the threshold.
            ids_keep = (ious <= threshold).nonzero().squeeze() # [m-1, ]. Because `nonzero()` adds extra dimension, squeeze it.
            if ids_keep.numel() == 0:
                break # If no box left, break.
            ids_sorted = ids_sorted[ids_keep+1] # `+1` is needed because `ids_sorted[0] = i`.

        return torch.LongTensor(ids)


if __name__ == '__main__':
    image_path = 'zhupipi.jpg'
    out_path = 'output.png'
    model_path = 'results/Mar05_16-34-45/model_best.pth'
    gpu_id = 0
    yolo = YOLODetector(model_path, gpu_id=gpu_id, conf_thresh=0.2, prob_thresh=0.1, nms_thresh=0.1)
    
    boxes, class_names, probs = yolo.detect(image_path)
    print(boxes, class_names, probs)
    if len(boxes):
        image = cv2.imread(image_path)
        image_boxes = visualize_boxes(image, boxes, class_names, probs)
        cv2.imwrite(out_path, image_boxes)