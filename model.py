import torch.nn as nn
from torchvision import models


class MyModel(nn.Module):
    def __init__(self, num_classes):
        super(MyModel, self).__init__()

        self.features = nn.Sequential(*list(models.resnet50(pretrained = True).children())[:-1])
        self.classifier = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        out = self.features(x)                
        out = out.reshape(out.size(0), -1) # performing flattening      
        out = self.classifier(out)
        return out