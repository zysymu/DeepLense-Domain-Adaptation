from torch import nn
from torchvision.models import resnet18, resnet50, wide_resnet50_2, resnext50_32x4d

class ResnetEncoder(nn.Module):
    def __init__(self, resnet, three_channels=False, pretrained=False, features_size=256):
        super(ResnetEncoder, self).__init__()
        
        if resnet == '18':
            self.resnet = resnet18(pretrained=pretrained)
        elif resnet == '50':
            self.resnet = resnet50(pretrained=pretrained)
        elif resnet == 'wide':
            self.resnet = wide_resnet50_2(pretrained=pretrained)
        elif resnet == 'x':
            self.resnet = resnext50_32x4d(pretrained=pretrained)

        if three_channels:
            self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        else:
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, features_size)

    def forward(self, x):
        x = self.resnet(x)
        return x

class ResnetClassifier(nn.Module):
    def __init__(self, features_size=256, num_classes=3):
        super(ResnetClassifier, self).__init__()
        
        self.fc = nn.Linear(features_size, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x
    
class Discriminator(nn.Module):
    def __init__(self, features_size=256, hidden_features=256):
        super(Discriminator, self).__init__()

        self.discrim = nn.Sequential(
            nn.Linear(features_size, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, 2)
        )

    def forward(self, x):
        x = self.discrim(x)
        return x