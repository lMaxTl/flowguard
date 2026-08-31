import torch.nn as nn
import torch.nn.functional as F

__all__ = ['lenet', 'lenet5']


class LeNet(nn.Module):
    """A simple MNIST network

    Source: https://github.com/pytorch/examples/blob/master/mnist/main.py
    """
    def __init__(self, num_classes=10, rot_semi=False, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, num_classes)
        # Rotation-prediction head for the S4L semi-supervised attack (see
        # defenses/utils/semi_losses.py::Rotation_Loss, which calls
        # model.rot_forward). Mirrors the rot_semi pattern already used by
        # the CIFAR VGG/ResNet models: a second head off the same shared
        # trunk, predicting one of 4 rotation classes instead of the digit
        # class. Off by default so ordinary (non-S4L) callers get the exact
        # same model as before.
        self.rot_semi = rot_semi
        if rot_semi:
            self.rot_classifier = nn.Linear(500, 4)

    def _features(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x):
        x = self._features(x)
        x = self.fc2(x)
        return x

    def rot_forward(self, x):
        assert self.rot_semi, "Have not specified semisupervised loss in LeNet!"
        x = self._features(x)
        x = self.rot_classifier(x)
        return x


def lenet(num_classes, **kwargs):
    return LeNet(num_classes, **kwargs)


class LeNet5(nn.Module):
    """LeNet-5 for 32x32 inputs (disguide-main/disguide/network/lenet.py)."""

    def __init__(self, num_classes=10, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def lenet5(num_classes=10, **kwargs):
    return LeNet5(num_classes, **kwargs)
