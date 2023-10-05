
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
import torch

class ResNet50_SSL(nn.Module):
    def __init__(self, projector, supervised_pretrained=None):
        super(ResNet50_SSL, self).__init__()
        print(projector, supervised_pretrained)

        # Backbone: ResNet50
        self.backbone = models.resnet50(pretrained = supervised_pretrained)
        self.backbone.fc = nn.Identity()

        # Projector
        sizes = [2048] + list(map(int, projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # Normalization layer for the representations
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, x):
        z = self.projector(self.backbone(x))
        return F.normalize(self.bn(z), dim = -1)
    
    



class ResNet_Model(torch.nn.Module):
    def __init__(self, version, pretrained=False, num_classes = 1, use_sigmoid_head = True):
        super(ResNet_Model, self).__init__()
        self.num_classes = num_classes
        self.model = None
        self.use_sigmoid_head = use_sigmoid_head
        self.num_ftrs = 0
        if 18 == version:
            self.model = models.resnet18(pretrained=pretrained)
        elif 50 == version:
            self.model = models.resnet50(pretrained=pretrained)
        elif 101 == version:
            self.model = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError("Select ResNet architecture from 18 | 50 | 101")
        self.num_ftrs=self.model.fc.in_features
        
        self.model.fc = nn.Sequential(nn.Dropout(0.8), nn.Linear(self.num_ftrs, 1))#(nn.Linear(self.num_ftrs, 256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, 1))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        output = None
        if True == self.use_sigmoid_head:
            output = self.sig(self.model(x))
        else:
            output = self.model(x)
        return output