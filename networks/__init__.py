
import torch
import torch.nn as nn
from torchvision import models

def load_model(name, outputsize, pretrained=None):

    if pretrained:
        pretrained = True
    else:
        pretrained = False

    if name.lower() in 'alexnet_micro':
        model = models.alexnet(pretrained=pretrained)
        model.features[0] = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3)
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = nn.Linear(256, outputsize)
    elif name.lower() in 'resnet18_micro':
        model = models.resnet18(pretrained=pretrained)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        model.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        model.fc = nn.Linear(model.fc.in_features, outputsize)
    elif name.lower() in 'densenet121_micro':
        model = models.densenet121(pretrained=pretrained)
        model.features.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        model.features.pool0 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        model.classifier = nn.Linear(model.classifier.in_features, outputsize)

    return model



