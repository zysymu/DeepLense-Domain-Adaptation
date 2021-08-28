import geffnet
from torch import nn

class Encoder(nn.Module):
    def __init__(self, num=4, three_channels=False, features_size=512):
        """
        EfficientNet based neural network that receives images and encodes them into an array of size `features_size`.
        Arguments:
        ----------
        num: int [0, 2, 4, 6]
            Kind of ResNet to be used as a backbone.
        three_channels: bool
            If True enables the network to receive three-channel images.
        features_size: int
            Size of encoded features array.
        """

        super(Encoder, self).__init__()

        if num == 0:
            self.effnet = geffnet.efficientnet_b0()
        if num == 2:
            self.effnet = geffnet.efficientnet_b2()
        if num == 4:
            self.effnet = geffnet.efficientnet_b4()
        if num == 6:
            self.effnet = geffnet.efficientnet_b6()

        if not three_channels:
            self.effnet.conv_stem = nn.Conv2d(1, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        self.effnet.classifier = nn.Linear(self.effnet.classifier.in_features, features_size, bias=True)

    def forward(self, x):
        x = self.effnet(x)
        return x

class Classifier(nn.Module):
    def __init__(self, features_size=512, num_classes=3):
        """
        Neural network that receives an array of size `features_size` and classifies it into `num_classes` classes.
        Arguments:
        ----------
        features_size: int
            Size of encoded features array.
        num_classes: int
            Number of classes to classify the encoded array into.
        """

        super(Classifier, self).__init__()
        self.fc = nn.Linear(features_size, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x