import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50, ResNet50_Weights

class YOLOv1(nn.Module):
    def __init__(self, feature_size = 7, num_bboxes=2, num_classes=20, bn=True):
        super(YOLOv1, self).__init__()
        self.S = feature_size
        self.B = num_bboxes
        self.C = num_classes
        backbone = resnet50(weights = ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.dimension_deduce_conv_layer = nn.Sequential(
                nn.Conv2d(2048, 1024, 3, padding=1),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1, inplace=True)
            )

        self.head = nn.Sequential(
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5, inplace=False), # is it okay to use Dropout with BatchNorm?
            nn.Linear(4096, self.S * self.S * (5 * self.B + self.C)),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.features(x)
        x = self.dimension_deduce_conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        x = x.view(-1, self.S, self.S, 5 * self.B + self.C)
        return x


if __name__ == '__main__':
    yolo = YOLOv1()
    x = torch.rand(8, 3, 224, 224)
    print(yolo(x).shape)
